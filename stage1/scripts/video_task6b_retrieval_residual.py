"""
Retrieval Residual — v1.7 multi-dim user KNN retrieval as a component.

Builds on the user's mvp_v1_4 → v1_7 evolution:
  - Multi-dim user embedding via PCA over 7 user count features
  - KNN retrieval K=50, self-excluded
  - Rich statistical features (mean/std/percentiles/weighted/dist)
  - CatBoost predicts log-residual from retrieval features
  - Output: residual component CSV ready for overlay blending
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Path to the best-model package scripts for shared utilities
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from video_task6b_residual_lab import read_submission, write_submission


def mape(y_true, y_pred):
    y_true = np.maximum(np.abs(y_true), 1e-8)
    return np.mean(np.abs((y_true - y_pred) / y_true))

DATA_DIR = Path("data")
RETRIEVAL_K = 50

USER_EMB_COLS = [
    "user_follower_count_log",
    "user_likes_count_log",
    "user_video_count_log",
    "user_digg_count_log",
    "user_friend_count_log",
    "user_following_count_log",
    "user_heart_count_log",
]

COUNT_COLS = [
    "user_following_count", "user_follower_count", "user_likes_count",
    "user_video_count", "user_digg_count", "user_heart_count", "user_friend_count",
]

# Time-based splits for honest CV
SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}


def load_data():
    posts_train = pd.read_parquet(DATA_DIR / "posts_train.parquet")
    posts_test = pd.read_parquet(DATA_DIR / "posts_test.parquet")
    users_train = pd.read_parquet(DATA_DIR / "users_train.parquet")
    users_test = pd.read_parquet(DATA_DIR / "users_test.parquet")
    videos_train = pd.read_parquet(DATA_DIR / "videos_train.parquet")
    videos_test = pd.read_parquet(DATA_DIR / "videos_test.parquet")
    labels_train = pd.read_parquet(DATA_DIR / "labels_train.parquet")

    posts = pd.concat([posts_train, posts_test], ignore_index=True)
    users = pd.concat([users_train, users_test], ignore_index=True)
    videos = pd.concat([videos_train, videos_test], ignore_index=True)

    return posts, users, videos, labels_train


def build_retrieval_features(users_df, labels_train):
    """v1.7-style multi-dim retrieval with rich statistics."""
    user_feat = users_df.drop_duplicates(subset="uid", keep="first").copy()

    for col in COUNT_COLS:
        user_feat[col + "_log"] = np.log1p(user_feat[col])

    global_mean = float(labels_train["popularity"].mean())

    emb_data = user_feat[USER_EMB_COLS].fillna(0).values.astype(np.float64)
    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(emb_data)
    n_pca = min(8, len(USER_EMB_COLS))
    pca = PCA(n_components=n_pca, random_state=42)
    user_emb = pca.fit_transform(emb_scaled).astype(np.float32)

    user_label = labels_train.groupby("uid")["popularity"].mean()
    labeled_uids = set(user_label.index)
    all_uids = user_feat["uid"].tolist()

    labeled_mask = np.array([uid in labeled_uids for uid in all_uids])
    labeled_idx = np.where(labeled_mask)[0]
    labeled_emb = user_emb[labeled_idx]
    labeled_uid_arr = np.array([all_uids[i] for i in labeled_idx])

    query_k = min(RETRIEVAL_K + 1, len(labeled_idx))
    nn = NearestNeighbors(n_neighbors=query_k, metric="euclidean", n_jobs=-1)
    nn.fit(labeled_emb)
    distances, neighbor_idx = nn.kneighbors(user_emb)

    all_labels = []
    all_dists = []
    for i, (nbrs, dists) in enumerate(zip(neighbor_idx, distances)):
        uid = all_uids[i]
        row_labels = []
        row_dists = []
        for j, d in zip(nbrs, dists):
            nbr_uid = labeled_uid_arr[j]
            if nbr_uid == uid:
                continue
            lbl = user_label.get(nbr_uid, np.nan)
            row_labels.append(lbl if not np.isnan(lbl) else global_mean)
            row_dists.append(d)
            if len(row_labels) >= RETRIEVAL_K:
                break
        while len(row_labels) < RETRIEVAL_K:
            row_labels.append(global_mean)
            row_dists.append(1e6)
        all_labels.append(row_labels[:RETRIEVAL_K])
        all_dists.append(row_dists[:RETRIEVAL_K])

    L = np.array(all_labels, dtype=np.float32)
    D = np.clip(np.array(all_dists, dtype=np.float32), 1e-3, None)

    # Rich statistics
    feat = pd.DataFrame(index=user_feat.index)
    feat["retrieval_mean"] = L.mean(axis=1)
    feat["retrieval_std"] = L.std(axis=1)
    feat["retrieval_max"] = L.max(axis=1)
    feat["retrieval_min"] = L.min(axis=1)
    feat["retrieval_median"] = np.median(L, axis=1)

    L_sorted = np.sort(L, axis=1)
    feat["retrieval_top3_mean"] = L_sorted[:, -3:].mean(axis=1)
    feat["retrieval_top5_mean"] = L_sorted[:, -5:].mean(axis=1)

    w = 1.0 / D
    w_sum = w.sum(axis=1)
    feat["retrieval_weighted_mean"] = (w * L).sum(axis=1) / w_sum
    wmean = feat["retrieval_weighted_mean"].values.reshape(-1, 1)
    feat["retrieval_weighted_std"] = np.sqrt((w * (L - wmean) ** 2).sum(axis=1) / w_sum)

    sim = np.exp(-D)
    feat["retrieval_sim_mean"] = sim.mean(axis=1)
    feat["retrieval_sim_max"] = sim.max(axis=1)
    feat["retrieval_sim_weighted"] = (sim * L).sum(axis=1) / (sim.sum(axis=1) + 1e-8)

    for p_name, p_val in [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90)]:
        feat[f"retrieval_{p_name}"] = np.percentile(L, p_val, axis=1).astype(np.float32)

    feat["retrieval_dist_mean"] = D.mean(axis=1)
    feat["retrieval_dist_std"] = D.std(axis=1)
    feat["retrieval_dist_min"] = D.min(axis=1)

    feat["uid"] = user_feat["uid"].values
    return feat


def build_per_pid_features(retrieval_feat, posts_df):
    """Map user-level retrieval features to PID level."""
    pid_to_uid = dict(zip(posts_df["pid"], posts_df["uid"]))
    uid_to_idx = {uid: i for i, uid in enumerate(retrieval_feat["uid"])}
    feat_cols = [c for c in retrieval_feat.columns if c != "uid"]

    rows = []
    for pid in posts_df["pid"]:
        uid = pid_to_uid.get(pid)
        idx = uid_to_idx.get(uid)
        if idx is not None:
            rows.append({c: retrieval_feat[c].iloc[idx] for c in feat_cols})
        else:
            rows.append({c: np.nan for c in feat_cols})

    result = pd.DataFrame(rows, index=posts_df["pid"])
    result = result.fillna(result.median())
    return result


def catboost_residual_fit_predict(
    X_train, y_train, X_target, depth=5, l2=20, iterations=600
):
    """Train CatBoost on retrieval features to predict log residual."""
    params = {
        "loss_function": "MAE",
        "iterations": iterations,
        "learning_rate": 0.03,
        "depth": depth,
        "l2_leaf_reg": l2,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    models = []
    for train_idx, val_idx in kf.split(X_train):
        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X_train.iloc[train_idx], y_train.iloc[train_idx]),
            eval_set=Pool(X_train.iloc[val_idx], y_train.iloc[val_idx]),
        )
        oof[val_idx] = model.predict(X_train.iloc[val_idx])
        models.append(model)

    test_preds = np.zeros(len(X_target))
    for model in models:
        test_preds += model.predict(X_target)
    test_preds /= len(models)

    return oof, test_preds, models


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="v1.7 retrieval residual component for SMP video task 6b."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--shrinks", default="0.02,0.04,0.06,0.08,0.10,0.15")
    parser.add_argument("--depths", default="4,5,6")
    parser.add_argument("--l2s", default="10,20,50")
    parser.add_argument("--name-prefix", default="candidate_retrieval_resid")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--component-dir")
    parser.add_argument("--report")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)

    print("=" * 60)
    print("Retrieval Residual: v1.7 multi-dim KNN user retrieval")
    print(f"  K={RETRIEVAL_K}, embedding={USER_EMB_COLS}")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading data...")
    posts, users, videos, labels_train = load_data()
    print(f"  Posts: {len(posts)}, Users: {users['uid'].nunique()}")

    # Build retrieval features
    print("\n[2/4] Building retrieval features (v1.7 rich stats)...")
    retrieval_feat = build_retrieval_features(users, labels_train)
    print(f"  Retrieval feature dims: {retrieval_feat.shape[1] - 1}")  # minus uid col

    per_pid = build_per_pid_features(retrieval_feat, posts)

    # Identify train/test PIDs
    train_pids = labels_train["pid"].tolist()
    test_pids_all = [p for p in posts["pid"] if p not in set(train_pids)]
    label_map = dict(zip(labels_train["pid"], labels_train["popularity"]))

    # Training data: retrieval features + log popularity as target
    X_train = per_pid.loc[train_pids]
    y_train = pd.Series({p: np.log(np.clip(label_map[p], 1e-6, None)) for p in train_pids})

    # Grid search over depths/l2s with time-based CV
    print("\n[3/4] Grid search for best CatBoost params...")
    shrinks = parse_float_list(args.shrinks)
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    l2s = parse_float_list(args.l2s)

    posts["month_key"] = pd.to_datetime(posts["post_time"]).dt.strftime("%Y-%m")
    pid_to_month = dict(zip(posts["pid"], posts["month_key"]))

    rows = []
    for depth in depths:
        for l2 in l2s:
            gains = []
            for split_name, (hist_months, val_months) in SPLITS.items():
                hist_pids = [p for p in train_pids if pid_to_month.get(p, "") in hist_months]
                val_pids = [p for p in train_pids if pid_to_month.get(p, "") in val_months]
                if len(hist_pids) < 30 or len(val_pids) < 10:
                    continue
                X_hist = X_train.loc[hist_pids]
                X_val = X_train.loc[val_pids]
                y_hist = y_train.loc[hist_pids]
                y_val = y_train.loc[val_pids]

                _, val_logpop, _ = catboost_residual_fit_predict(
                    X_hist, y_hist, X_val, depth=depth, l2=l2
                )
                val_pred = np.exp(np.clip(val_logpop, -5.0, 5.0))
                target_val = np.array([label_map[p] for p in val_pids])
                val_mape = mape(target_val, val_pred)
                gains.append(-val_mape)

            if gains:
                mapes = [-g for g in gains]
                rows.append({
                    "depth": depth,
                    "l2": l2,
                    "mean_mape": float(np.mean(mapes)),
                    "min_mape": float(np.min(mapes)),
                })

    report = pd.DataFrame(rows).sort_values(["mean_mape", "min_mape"], ascending=True)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        print(f"  Report saved to {report_path}")

    print(report.head(10).to_string(index=False))

    # Train on full training data with best params
    print("\n[4/4] Training final retrieval model and generating residual...")
    X_test = per_pid.loc[test_pids_all]
    best_row = report.iloc[0]
    oof_train_logpop, test_logpop, models = catboost_residual_fit_predict(
        X_train, y_train, X_test,
        depth=int(best_row["depth"]),
        l2=float(best_row["l2"]),
    )

    # Load base submission (test PIDs only) and compute retrieval residual
    base = read_submission(args.base_submission)
    test_logpop_series = pd.Series(
        np.clip(test_logpop, -5.0, 5.0), index=X_test.index
    )
    retrieval_base = pd.Series(
        np.exp(test_logpop_series.values), index=test_logpop_series.index
    )

    # Align with base submission
    common_pids = base.index.intersection(retrieval_base.index)
    base_test = base.loc[common_pids]
    retrieval_base = retrieval_base.loc[common_pids]

    # Log residual: how retrieval prediction differs from base submission
    retrieval_log_resid = np.log(
        np.clip(retrieval_base.to_numpy(float), 1e-6, None)
        / np.clip(base_test.to_numpy(float), 1e-6, None)
    )

    # Write outputs at multiple shrinks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for shrink in shrinks:
        pred = base.to_numpy(float).copy()
        test_pred = base_test.to_numpy(float) * np.exp(
            np.clip(shrink * retrieval_log_resid, -1.0, 1.0)
        )
        idx_map = {p: i for i, p in enumerate(base.index)}
        for j, p in enumerate(base_test.index):
            pred[idx_map[p]] = test_pred[j]
        label = str(shrink).replace(".", "p")
        path = out_dir / f"{args.name_prefix}_s{label}.csv"
        write_submission(base.index, pred, path)
        written.append(str(path))

    # Save raw retrieval prediction as a component CSV
    if args.component_dir:
        comp_dir = Path(args.component_dir)
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_path = comp_dir / f"{args.name_prefix}_raw_prediction.csv"
        comp_df = pd.DataFrame({
            "pid": retrieval_base.index,
            "polularity_score": retrieval_base.values,
        })
        comp_df.to_csv(comp_path, index=False)
        print(f"  Component saved to {comp_path}")

    meta = {
        "base_submission": args.base_submission,
        "retrieval_k": RETRIEVAL_K,
        "embedding_cols": USER_EMB_COLS,
        "pca_components": min(8, len(USER_EMB_COLS)),
        "best_params": {
            "depth": int(best_row["depth"]),
            "l2": float(best_row["l2"]),
        },
        "cv_report": report.head(5).to_dict("records"),
        "oof_train_logpop_stats": {
            "mean": float(np.mean(oof_train_logpop)),
            "std": float(np.std(oof_train_logpop)),
        },
        "retrieval_log_resid_stats": {
            "mean": float(np.mean(retrieval_log_resid)),
            "std": float(np.std(retrieval_log_resid)),
            "mad": float(np.mean(np.abs(retrieval_log_resid))),
        },
        "written": written,
    }

    meta_path = out_dir / f"{args.name_prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n  Meta saved to {meta_path}")
    print(f"  Best params: depth={best_row['depth']}, l2={best_row['l2']}")
    print("  Written candidates:")
    for path in written:
        print(f"    {path}")


if __name__ == "__main__":
    main()
