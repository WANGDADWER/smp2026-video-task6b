"""
Reconstructed: video_task6b_lgbm_dense_residual.py
Provides add_basic_dense_features(train, test) for torch_fusion.py.
Adds normalized dense numeric features: user stats, video properties,
interaction features, and bucketized features.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_basic_dense_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Add basic dense numeric features to train and test DataFrames.
    Returns (train, test) with additional columns.
    """
    # Standardize key numeric columns
    numeric_patterns = [
        "log1p_", "likes_per_", "heart_per_", "followers_per_",
        "following_follower", "follower_per_", "digg_per_",
        "pixel_count", "aspect_ratio", "duration", "fps",
        "width", "height", "file_size",
    ]

    numeric_cols = []
    for col in train.columns:
        if col in ("pid", "uid", "vid", "post_time", "dt", "split",
                     "popularity", "oof_pred", "month_key", "video_path",
                     "post_content", "post_suggested_words"):
            continue
        if any(pat in col for pat in numeric_patterns):
            numeric_cols.append(col)

    # Add all remaining numeric columns
    for col in train.columns:
        if col not in numeric_cols and pd.api.types.is_numeric_dtype(train[col]):
            if col not in ("pid", "uid", "vid"):
                numeric_cols.append(col)

    numeric_cols = list(dict.fromkeys(numeric_cols))  # deduplicate preserving order
    numeric_cols = [c for c in numeric_cols if c in train.columns and c in test.columns]

    if not numeric_cols:
        return train, test

    # Standardize: use only columns that are numeric in BOTH train and test
    safe_cols = [c for c in numeric_cols
                 if c in train.columns and c in test.columns
                 and pd.api.types.is_numeric_dtype(train[c])
                 and pd.api.types.is_numeric_dtype(test[c])]
    if not safe_cols:
        return train, test
    ref = train[safe_cols]
    med = ref.median().fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(ref.fillna(med))

    def _add_dense(df, name_prefix):
        x = df[safe_cols].fillna(med).to_numpy(np.float32)
        x = np.where(np.isfinite(x), x, 0.0)
        x = scaler.transform(x).astype(np.float32)
        for i in range(min(x.shape[1], 128)):
            df[f"{name_prefix}_dense_{i}"] = x[:, i]
        return df

    train = _add_dense(train, "dense")
    test = _add_dense(test, "dense")
    return train, test
