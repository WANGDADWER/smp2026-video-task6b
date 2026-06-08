from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape
from video_task6b_cat_residual import add_features, attach_feature_csvs
from video_task6b_residual_lab import read_submission, write_submission


SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}

CAT_COLS = [
    "post_location",
    "post_text_language",
    "video_ratio",
    "music_title",
    "month_str",
    "hour_str",
    "dow_str",
    "duration_bucket",
    "music_duration_bucket",
    "followers_bucket",
    "uid_freq_bucket",
    "asr_language",
    "asr_present",
    "ocr_present",
]

CORE_NUMERIC = [
    "pid_num",
    "vid_num",
    "uid_num",
    "pid_vid_gap",
    "pid_uid_gap",
    "vid_uid_gap",
    "pid_per_uid",
    "vid_per_uid",
    "month_index",
    "day_index",
    "hour",
    "dow",
    "day",
    "is_weekend",
    "post_content_len",
    "post_content_tokens",
    "post_content_unique_tokens",
    "suggested_text_len",
    "suggested_text_tokens",
    "suggested_text_unique_tokens",
    "full_text_len",
    "full_text_tokens",
    "full_text_unique_tokens",
    "suggested_count",
    "is_original_sound",
    "log1p_user_following_count",
    "log1p_user_follower_count",
    "log1p_user_likes_count_fixed",
    "log1p_user_video_count",
    "log1p_user_digg_count",
    "log1p_user_heart_count_fixed",
    "log1p_user_friend_count",
    "video_duration",
    "music_duration",
    "aspect_ratio",
    "log1p_pixel_count",
    "music_to_video_duration",
    "duration_gap",
    "log1p_likes_per_video",
    "log1p_heart_per_video",
    "log1p_followers_per_video",
    "log1p_heart_per_follower",
    "log1p_likes_per_follower",
    "following_follower_ratio",
    "follower_per_following",
]


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_name(path: str) -> str:
    stem = Path(path).stem.lower()
    for token in ["outputs", "features", "frame8", "all", "caption", "ocr", "asr"]:
        stem = stem.replace(token, "")
    stem = stem.replace("__", "_").strip("_")
    return stem[:36]


def embedding_columns(path: str, max_raw_dims: int = 0) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    skip = {"pid", "uid", "vid", "split", "video_path", "video_abspath"}
    cols = [c for c in header if c not in skip]
    emb_like = [
        c
        for c in cols
        if any(token in c for token in ["emb_", "_mean_", "_cls_", "_std_"])
        and not c.endswith("_ok")
        and not c.endswith("_n_frames")
    ]
    if not emb_like:
        emb_like = cols
    if max_raw_dims and len(emb_like) > max_raw_dims:
        emb_like = emb_like[:max_raw_dims]
    return ["pid"] + emb_like


def load_embedding_matrix(
    path: str,
    train_pids: pd.Series,
    test_pids: pd.Series,
    n_components: int,
    seed: int,
    max_raw_dims: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    usecols = embedding_columns(path, max_raw_dims=max_raw_dims)
    feat = pd.read_csv(path, usecols=usecols)
    feat = feat.drop_duplicates("pid")
    all_pids = pd.concat([train_pids, test_pids], ignore_index=True).to_frame("pid")
    x = all_pids.merge(feat, on="pid", how="left")
    cols = [c for c in x.columns if c != "pid"]
    arr = x[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(arr))
    if len(inds[0]):
        arr[inds] = med[inds[1]]
    arr = normalize(arr, norm="l2", axis=1, copy=False)
    if n_components and arr.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
        arr = pca.fit_transform(arr).astype(np.float32)
        arr = normalize(arr, norm="l2", axis=1, copy=False)
        explained = float(np.sum(pca.explained_variance_ratio_))
    else:
        explained = 1.0
    n_train = len(train_pids)
    meta = {
        "path": path,
        "raw_dims": len(cols),
        "dims": int(arr.shape[1]),
        "pca_explained": explained,
        "missing_rows": int(x[cols[0]].isna().sum()) if cols else 0,
    }
    return arr[:n_train], arr[n_train:], meta


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = np.sum(weights, axis=1) + 1e-9
    return np.sum(values * weights, axis=1) / denom


def knn_features(
    ref_df: pd.DataFrame,
    query_df: pd.DataFrame,
    ref_emb: np.ndarray,
    query_emb: np.ndarray,
    ks: list[int],
    prefix: str,
    exclude_self: bool,
) -> pd.DataFrame:
    max_k = max(ks)
    n_neighbors = min(len(ref_df), max_k + (1 if exclude_self else 0))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(ref_emb)
    dist, idx = nn.kneighbors(query_emb, return_distance=True)
    sim = 1.0 - dist
    if exclude_self:
        ref_pid = ref_df["pid"].astype(str).to_numpy()
        query_pid = query_df["pid"].astype(str).to_numpy()
        max_k = max(ks)
        clean_idx = np.empty((len(query_df), max_k), dtype=int)
        clean_sim = np.empty((len(query_df), max_k), dtype=float)
        for row_i in range(len(query_df)):
            keep = ref_pid[idx[row_i]] != query_pid[row_i]
            row_idx = idx[row_i][keep]
            row_sim = sim[row_i][keep]
            if len(row_idx) == 0:
                row_idx = idx[row_i][-1:]
                row_sim = sim[row_i][-1:]
            if len(row_idx) < max_k:
                pad = max_k - len(row_idx)
                row_idx = np.concatenate([row_idx, np.repeat(row_idx[-1], pad)])
                row_sim = np.concatenate([row_sim, np.repeat(row_sim[-1], pad)])
            clean_idx[row_i] = row_idx[:max_k]
            clean_sim[row_i] = row_sim[:max_k]
        idx = clean_idx
        sim = clean_sim

    ref_pop = ref_df["popularity"].to_numpy(float)
    ref_oof = np.clip(ref_df["oof_pred"].to_numpy(float), 1e-6, None)
    ref_resid = np.log(np.clip(ref_pop, 1e-6, None) / ref_oof)
    ref_logpop = np.log(np.clip(ref_pop, 1e-6, None))
    ref_low = (ref_pop <= 7.0).astype(float)
    ref_high = (ref_pop >= 11.0).astype(float)
    ref_dt = ref_df["dt"].to_numpy("datetime64[D]")
    query_dt = query_df["dt"].to_numpy("datetime64[D]")
    query_anchor = np.clip(query_df["anchor_for_proto"].to_numpy(float), 1e-6, None)

    rows: dict[str, np.ndarray] = {}
    for k in ks:
        kk = min(k, idx.shape[1])
        cur_idx = idx[:, :kk]
        cur_sim = sim[:, :kk]
        weights = np.power(np.clip(cur_sim, 0.0, None) + 1e-4, 2.0)
        ages = (query_dt[:, None] - ref_dt[cur_idx]).astype("timedelta64[D]").astype(float)
        ages = np.clip(ages, 0.0, 1000.0)
        time_weights = weights * np.exp(-ages / 120.0)

        pop_vals = ref_pop[cur_idx]
        log_vals = ref_logpop[cur_idx]
        resid_vals = ref_resid[cur_idx]
        low_vals = ref_low[cur_idx]
        high_vals = ref_high[cur_idx]

        pop_mean = weighted_mean(pop_vals, weights)
        pop_time = weighted_mean(pop_vals, time_weights)
        resid_mean = weighted_mean(resid_vals, weights)
        resid_time = weighted_mean(resid_vals, time_weights)
        log_mean = weighted_mean(log_vals, weights)

        base = f"{prefix}_k{kk}"
        rows[f"{base}_sim_max"] = cur_sim[:, 0]
        rows[f"{base}_sim_mean"] = np.mean(cur_sim, axis=1)
        rows[f"{base}_age_mean"] = np.mean(ages, axis=1)
        rows[f"{base}_pop_mean"] = pop_mean
        rows[f"{base}_pop_time_mean"] = pop_time
        rows[f"{base}_logpop_mean"] = log_mean
        rows[f"{base}_resid_mean"] = resid_mean
        rows[f"{base}_resid_time_mean"] = resid_time
        rows[f"{base}_top1_pop"] = pop_vals[:, 0]
        rows[f"{base}_top1_resid"] = resid_vals[:, 0]
        rows[f"{base}_low_rate"] = weighted_mean(low_vals, weights)
        rows[f"{base}_high_rate"] = weighted_mean(high_vals, weights)
        rows[f"{base}_log_pop_vs_anchor"] = np.log(np.clip(pop_mean, 1e-6, None) / query_anchor)
        rows[f"{base}_time_log_pop_vs_anchor"] = np.log(np.clip(pop_time, 1e-6, None) / query_anchor)
    return pd.DataFrame(rows, index=query_df.index)


def core_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORE_NUMERIC if c in df.columns]
    cats = [c for c in CAT_COLS if c in df.columns]
    out = df[["pid"] + cols + cats].copy()
    for col in cats:
        out[col] = out[col].fillna("__NA__").astype(str)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_proto_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    embeddings: list[tuple[str, np.ndarray, np.ndarray]],
    ks: list[int],
    mode: str,
    hist_months: list[str] | None = None,
    val_months: list[str] | None = None,
    full_ref_months: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "split":
        assert hist_months is not None and val_months is not None
        hist = train[train["month_key"].isin(hist_months)].copy()
        val = train[train["month_key"].isin(val_months)].copy()
        hist_idx = hist.index.to_numpy()
        val_idx = val.index.to_numpy()
        hist_parts = [core_frame(hist).set_index(hist.index)]
        val_parts = [core_frame(val).set_index(val.index)]
        for name, tr_emb, _te_emb in embeddings:
            hist_parts.append(
                knn_features(hist, hist, tr_emb[hist_idx], tr_emb[hist_idx], ks, name, exclude_self=True)
            )
            val_parts.append(
                knn_features(hist, val, tr_emb[hist_idx], tr_emb[val_idx], ks, name, exclude_self=False)
            )
        return pd.concat(hist_parts, axis=1), pd.concat(val_parts, axis=1)

    if mode == "full":
        ref_train = train if not full_ref_months else train[train["month_key"].isin(full_ref_months)].copy()
        ref_idx = ref_train.index.to_numpy()
        train_parts = [core_frame(train).set_index(train.index)]
        test_parts = [core_frame(test).set_index(test.index)]
        for name, tr_emb, te_emb in embeddings:
            train_parts.append(
                knn_features(ref_train, train, tr_emb[ref_idx], tr_emb, ks, name, exclude_self=True)
            )
            test_parts.append(
                knn_features(ref_train, test, tr_emb[ref_idx], te_emb, ks, name, exclude_self=False)
            )
        return pd.concat(train_parts, axis=1), pd.concat(test_parts, axis=1)

    raise ValueError(mode)


def model_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    skip = {"pid"}
    cats = [c for c in CAT_COLS if c in df.columns]
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        if col in cats or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols, cats


def make_pool(df: pd.DataFrame, cols: list[str], cats: list[str], reference: pd.DataFrame, label=None) -> Pool:
    out = df[cols].copy()
    for col in cols:
        if col in cats:
            out[col] = out[col].fillna("__NA__").astype(str)
        else:
            med = pd.to_numeric(reference[col], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(med)
    cat_idx = [cols.index(c) for c in cats]
    return Pool(out, label=label, cat_features=cat_idx)


def fit_predict(train_x, valid_x, y, cols, cats, args) -> np.ndarray:
    model = CatBoostRegressor(
        loss_function=args.loss,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(make_pool(train_x, cols, cats, train_x, y))
    return model.predict(make_pool(valid_x, cols, cats, train_x))


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    b = basis - float(np.mean(basis))
    vv = v - float(np.mean(v))
    denom = float(np.dot(b, b))
    if denom <= 1e-12:
        return v
    beta = float(np.dot(vv, b) / denom)
    return v - beta * basis


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype retrieval residual model for SMP video task 6b.")
    parser.add_argument("--data-dir", default="video-data")
    parser.add_argument("--oof", default="outputs/oof/power_tabular_v1_fix_oof.csv")
    parser.add_argument("--oof-pred-col", default="pred_blend")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--embedding-csvs", nargs="+", required=True)
    parser.add_argument("--feature-csvs", nargs="*", default=["outputs/features/video_asr.csv", "outputs/features/frame_ocr_rapid.csv"])
    parser.add_argument("--ks", default="3,10,25,50")
    parser.add_argument("--full-ref-months", default="")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--max-raw-dims", type=int, default=0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--l2", type=float, default=25.0)
    parser.add_argument("--loss", default="MAE")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--shrinks", default="0.02,0.035,0.05,0.08,0.12")
    parser.add_argument("--clip-log", type=float, default=0.8)
    parser.add_argument("--orth-anchor")
    parser.add_argument("--orth-source")
    parser.add_argument("--name-prefix", default="candidate_250_proto_resid")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/prototype_residual_report.csv")
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    train, test = load_train_test(args.data_dir)
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)
    if args.feature_csvs:
        train, test = attach_feature_csvs(train, test, args.feature_csvs)
    train, test = add_features(train, test)
    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    test["month_key"] = test["dt"].dt.strftime("%Y-%m")

    oof = pd.read_csv(args.oof)
    train = train.merge(
        oof[["pid", args.oof_pred_col]].rename(columns={args.oof_pred_col: "oof_pred"}),
        on="pid",
        how="inner",
    )
    train["oof_pred"] = np.clip(pd.to_numeric(train["oof_pred"], errors="coerce").to_numpy(float), 1e-6, None)
    train["anchor_for_proto"] = train["oof_pred"]
    base = read_submission(args.base_submission)
    test = test.merge(base.rename("base_pred"), left_on="pid", right_index=True, how="inner")
    test["anchor_for_proto"] = np.clip(test["base_pred"].to_numpy(float), 1e-6, None)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    embeddings = []
    emb_meta = []
    for path in args.embedding_csvs:
        name = safe_name(path)
        tr_emb, te_emb, meta = load_embedding_matrix(
            path,
            train["pid"],
            test["pid"],
            args.pca_components,
            args.seed,
            args.max_raw_dims,
        )
        embeddings.append((name, tr_emb, te_emb))
        emb_meta.append({"name": name, **meta})

    ks = parse_int_list(args.ks)
    shrinks = parse_float_list(args.shrinks)
    rows = []
    for split_name, (hist_months, val_months) in SPLITS.items():
        hist_x, val_x = build_proto_matrix(
            train,
            test,
            embeddings,
            ks,
            mode="split",
            hist_months=hist_months,
            val_months=val_months,
        )
        hist = train.loc[hist_x.index].copy()
        val = train.loc[val_x.index].copy()
        cols, cats = model_columns(hist_x)
        y = np.log(np.clip(hist["popularity"].to_numpy(float), 1e-6, None) / hist["oof_pred"].to_numpy(float))
        pred_log = np.clip(fit_predict(hist_x, val_x, y, cols, cats, args), -args.clip_log, args.clip_log)
        base_pred = val["oof_pred"].to_numpy(float)
        target = val["popularity"].to_numpy(float)
        base_mape = mape(target, base_pred)
        for shrink in shrinks:
            pred = base_pred * np.exp(shrink * pred_log)
            rows.append(
                {
                    "split": split_name,
                    "shrink": shrink,
                    "base_mape": base_mape,
                    "mape": mape(target, pred),
                    "gain": base_mape - mape(target, pred),
                    "mean_abs_log": float(np.mean(np.abs(pred_log))),
                    "features": len(cols),
                    "cats": len(cats),
                    "train_rows": len(hist_x),
                    "valid_rows": len(val_x),
                }
            )

    split_report = pd.DataFrame(rows)
    summary = (
        split_report.groupby("shrink")
        .agg(mean_gain=("gain", "mean"), min_gain=("gain", "min"), mean_mape=("mape", "mean"), mean_abs_log=("mean_abs_log", "mean"))
        .reset_index()
        .sort_values(["mean_gain", "min_gain"], ascending=False)
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    split_report.to_csv(report_path.with_name(report_path.stem + "_splits.csv"), index=False)
    summary.to_csv(report_path, index=False)

    full_ref_months = parse_list(args.full_ref_months)
    train_x, test_x = build_proto_matrix(train, test, embeddings, ks, mode="full", full_ref_months=full_ref_months)
    cols, cats = model_columns(train_x)
    y = np.log(np.clip(train["popularity"].to_numpy(float), 1e-6, None) / train["oof_pred"].to_numpy(float))
    pred_log = np.clip(fit_predict(train_x, test_x, y, cols, cats, args), -args.clip_log, args.clip_log)
    if args.orth_anchor and args.orth_source:
        anchor = read_submission(args.orth_anchor).loc[base.index]
        source = read_submission(args.orth_source).loc[base.index]
        basis = np.log(np.clip(source.to_numpy(float), 1e-6, None) / np.clip(anchor.to_numpy(float), 1e-6, None))
        pred_log = project_out(pred_log, basis)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base.loc[test["pid"]]
    written = []
    for shrink in shrinks:
        pred = base.to_numpy(float) * np.exp(shrink * pred_log)
        label = str(shrink).replace(".", "p").replace("-", "m")
        path = out_dir / f"{args.name_prefix}_s{label}.csv"
        write_submission(base.index, pred, path)
        written.append(str(path))

    meta = {
        "base_submission": args.base_submission,
        "embedding_meta": emb_meta,
        "feature_csvs": args.feature_csvs,
        "ks": ks,
        "full_ref_months": full_ref_months,
        "columns": cols,
        "cat_columns": cats,
        "written": written,
        "pred_log_stats": {
            "mean": float(np.mean(pred_log)),
            "std": float(np.std(pred_log)),
            "mad": float(np.mean(np.abs(pred_log))),
            "min": float(np.min(pred_log)),
            "max": float(np.max(pred_log)),
        },
    }
    report_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(summary.to_string(index=False))
    print("Prediction log residual stats:", meta["pred_log_stats"])
    print("Written candidates:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
