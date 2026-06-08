"""
Step 1: Train tabular CatBoost baseline with time-split validation.
Matches the original approach: time-split CV, uid as categorical, grid search.
Produces outputs/oof/power_tabular_v1_fix_oof.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import time

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape

DATA_DIR = "video-data"
OOF_PATH = Path("outputs/oof/power_tabular_v1_fix_oof.csv")
OOF_PATH.parent.mkdir(parents=True, exist_ok=True)

# Time-split validation matching the original approach
SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}

CAT_COLS = [
    "uid",
    "post_location",
    "post_text_language",
    "video_ratio",
    "video_format",
    "music_title",
    "month_str",
    "hour_str",
    "dow_str",
    "duration_bucket",
    "music_duration_bucket",
    "followers_bucket",
    "uid_freq_bucket",
    "pid_bin",
    "vid_bin",
    "uid_bin",
]

TEXT_COLS = [
    "content_text",
    "suggested_text",
    "full_text",
]


def add_features(train, test):
    """Re-implemented to avoid pandas type errors."""
    train = train.copy()
    test = test.copy()

    def id_number(s):
        return s.astype(str).str.extract(r"(\d+)")[0].astype(float)

    def bucketize(values, bins, labels):
        return pd.cut(pd.to_numeric(values, errors="coerce"),
                      bins=bins, labels=labels, include_lowest=True).astype(str)

    def qbucket(train_values, values, prefix, q=8):
        clean = pd.to_numeric(train_values, errors="coerce").dropna()
        if clean.empty:
            return pd.Series(f"{prefix}_all", index=values.index)
        edges = np.unique(np.nanquantile(clean, np.linspace(0, 1, q + 1)))
        if len(edges) <= 2:
            return pd.Series(f"{prefix}_all", index=values.index)
        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = [f"{prefix}_{i}" for i in range(len(edges) - 1)]
        return pd.cut(pd.to_numeric(values, errors="coerce"),
                      edges, labels=labels, include_lowest=True).astype(str)

    combined_uid_count = pd.concat([train["uid"], test["uid"]]).astype(str).value_counts()

    for frame in [train, test]:
        frame["pid_num"] = id_number(frame["pid"])
        frame["vid_num"] = id_number(frame["vid"])
        frame["uid_num"] = id_number(frame["uid"])
        frame["pid_vid_gap"] = frame["vid_num"] - frame["pid_num"]
        frame["pid_uid_gap"] = frame["pid_num"] - frame["uid_num"]
        frame["vid_uid_gap"] = frame["vid_num"] - frame["uid_num"]
        frame["pid_per_uid"] = frame["pid_num"] / (frame["uid_num"] + 1)
        frame["vid_per_uid"] = frame["vid_num"] / (frame["uid_num"] + 1)
        frame["month_index"] = frame["dt"].dt.year * 12 + frame["dt"].dt.month
        frame["day_index"] = (frame["dt"] - train["dt"].min()).dt.total_seconds().fillna(0) / 86400.0
        frame["month_str"] = frame["dt"].dt.month.fillna(0).astype(int).astype(str)
        frame["hour_str"] = frame["dt"].dt.hour.fillna(0).astype(int).astype(str)
        frame["dow_str"] = frame["dt"].dt.dayofweek.fillna(0).astype(int).astype(str)
        frame["duration_bucket"] = bucketize(
            frame["video_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["dur_10", "dur_30", "dur_60", "dur_120", "dur_120p"],
        )
        frame["music_duration_bucket"] = bucketize(
            frame["music_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["music_10", "music_30", "music_60", "music_120", "music_120p"],
        )
        frame["uid_freq_bucket"] = bucketize(
            frame["uid"].astype(str).map(combined_uid_count).fillna(1),
            [-np.inf, 1, 2, 4, np.inf],
            ["uid_1", "uid_2", "uid_4", "uid_5p"],
        )

    for source, name in [
        ("user_follower_count", "followers"),
        ("pid_num", "pid"),
        ("vid_num", "vid"),
        ("uid_num", "uid"),
    ]:
        train[f"{name}_bin"] = qbucket(train[source], train[source], name)
        test[f"{name}_bin"] = qbucket(train[source], test[source], name)

    for frame in [train, test]:
        frame["followers_bucket"] = frame.get("followers_bin", "followers_all").astype(str)

    # Replace inf in numeric columns only
    for frame in [train, test]:
        num_cols = frame.select_dtypes(include='number').columns
        frame[num_cols] = frame[num_cols].replace([np.inf, -np.inf], np.nan)

    return train, test


def model_columns(df, use_text_features=True):
    """Determine feature columns matching original approach."""
    skip = {
        "pid", "vid", "uid", "post_time", "video_path", "post_content",
        "post_suggested_words", "dt", "split", "popularity", "month_key",
        "content_text", "suggested_text", "full_text", "music_text",
        "asr_text", "ocr_text", "blip_caption",
    }
    cats = [c for c in CAT_COLS if c in df.columns and df[c].nunique() > 0]
    texts = [c for c in TEXT_COLS if use_text_features and c in df.columns and c not in cats]
    nums = []
    for col in df.columns:
        if col in skip or col in cats or col in texts:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            nums.append(col)
    return nums + cats + texts, cats, texts


def _make_pool(df, cols, cats, texts, reference=None, label=None):
    """Build CatBoost Pool matching original prepare_pool."""
    out = df[cols].copy()
    ref = reference if reference is not None else df
    for col in cols:
        if col in cats:
            out[col] = out[col].fillna("__NA__").astype(str)
        elif col in texts:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            med = pd.to_numeric(ref[col], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            out[col] = out[col].fillna(med)

    cols_present = [c for c in cols if c in out.columns]
    cat_idx = [cols_present.index(c) for c in cats if c in cols_present]
    text_idx = [cols_present.index(c) for c in texts if c in cols_present]
    return Pool(out[cols_present], label=label, cat_features=cat_idx, text_features=text_idx)


def fit_predict(train_df, target_df, cols, cats, texts, depth, l2, seed, loss, iterations):
    """Train a single CatBoost model and predict on target."""
    y = train_df["popularity"].to_numpy(float)
    train_pool = _make_pool(train_df, cols, cats, texts, label=y)
    target_pool = _make_pool(target_df, cols, cats, texts, reference=train_df)

    model = CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        learning_rate=0.035,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    model.fit(train_pool)
    return model.predict(target_pool), model


def main():
    print("Loading data...")
    train, test = load_train_test(DATA_DIR)
    train_min_time = train["post_time"].min()

    print("Building features...")
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)
    train, test = add_features(train, test)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    y = train["popularity"].to_numpy(float)

    cols, cats, texts = model_columns(train, use_text_features=True)
    # Preprocess text: replace commas with spaces so CatBoost tokenizer works on tags
    for col in texts:
        train[col] = train[col].fillna("").astype(str).str.replace(",", " ")
        test[col] = test[col].fillna("").astype(str).str.replace(",", " ")
    print(f"Features: {len(cols) - len(cats) - len(texts)} numeric + {len(cats)} cat + {len(texts)} text = {len(cols)}")
    print(f"Cat features: {cats}")
    print(f"Text features: {texts}")

    # Grid search over hyperparameters with time-split validation
    depths = [4, 6]
    l2s = [10, 30, 100]
    losses = ["MAE", "RMSE"]
    seed = 2026
    iterations = 700

    print("\nGrid search with time-split validation:")
    grid_rows = []
    best_config = None
    best_mean_gain = -999.0

    for depth in depths:
        for l2 in l2s:
            for loss in losses:
                gains = []
                for split_name, (hist_months, val_months) in SPLITS.items():
                    hist = train[train["month_key"].isin(hist_months)].copy()
                    val = train[train["month_key"].isin(val_months)].copy()
                    if len(hist) == 0 or len(val) == 0:
                        continue

                    t0 = time.time()
                    pred, _ = fit_predict(hist, val, cols, cats, texts, depth, l2, seed, loss, iterations)
                    elapsed = time.time() - t0

                    y_val = val["popularity"].to_numpy(float)
                    split_mape = mape(y_val, pred)
                    gains.append(split_mape)
                    print(f"  depth={depth} l2={l2} loss={loss:5s} split={split_name:7s} "
                          f"MAPE={split_mape:.4f} ({elapsed:.1f}s)")

                if gains:
                    mean_mape = float(np.mean(gains))
                    min_mape = float(np.min(gains))
                    mean_gain = -mean_mape  # Higher is better
                    grid_rows.append({
                        "depth": depth, "l2": l2, "loss": loss,
                        "mean_mape": mean_mape, "min_mape": min_mape,
                    })
                    if mean_gain > best_mean_gain:
                        best_mean_gain = mean_gain
                        best_config = (depth, l2, loss)
                    print(f"  => Mean MAPE={mean_mape:.4f}, Min MAPE={min_mape:.4f}")

    grid_df = pd.DataFrame(grid_rows).sort_values("mean_mape")
    print(f"\nGrid results:\n{grid_df.to_string(index=False)}")

    best_depth, best_l2, best_loss = best_config
    print(f"\nBest config: depth={best_depth}, l2={best_l2}, loss={best_loss}")

    # Full OOF predictions with time-split CV using best config
    print("\nGenerating OOF predictions with best config...")
    oof = np.zeros(len(train))
    for split_name, (hist_months, val_months) in SPLITS.items():
        hist = train[train["month_key"].isin(hist_months)]
        val = train[train["month_key"].isin(val_months)]
        if len(hist) == 0 or len(val) == 0:
            continue
        val_pred, _ = fit_predict(hist, val, cols, cats, texts,
                                  best_depth, best_l2, seed, best_loss, iterations)
        oof[val.index] = np.clip(val_pred, 1e-6, None)
        y_val = y[val.index]
        print(f"  Split {split_name}: MAPE={mape(y_val, oof[val.index]):.4f}")

    cv_mape = mape(y, oof)
    print(f"\nOverall CV MAPE: {cv_mape:.4f}")

    # Train final model on all data for test predictions
    print("\nTraining final model on all data...")
    test_preds, final_model = fit_predict(train, test, cols, cats, texts,
                                          best_depth, best_l2, seed, best_loss, iterations)
    test_preds = np.clip(test_preds, 1e-6, None)
    print(f"Test pred mean: {test_preds.mean():.4f}, std: {test_preds.std():.4f}")

    # Feature importance
    print("\nTop-20 Feature Importance:")
    feat_imp = sorted(zip(cols, final_model.get_feature_importance()),
                      key=lambda x: -x[1])
    for i, (name, imp) in enumerate(feat_imp[:20]):
        print(f"  {i+1:2d}. {name:40s} {imp:.4f}")

    # Save OOF
    oof_df = train[["pid"]].copy()
    oof_df["pred_blend"] = oof
    oof_df["popularity"] = y
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"\nSaved OOF to {OOF_PATH}")

    # Save submission
    base_sub = test[["pid"]].copy()
    base_sub["polularity_score"] = test_preds
    base_path = Path("outputs/submissions/candidate_tabular_baseline.csv")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_sub.to_csv(base_path, index=False)
    print(f"Saved baseline submission to {base_path}")


if __name__ == "__main__":
    main()
