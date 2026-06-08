"""
Build combined AMCFG anchor dense OCR/ASR features.
Merges: video_stats + file_props + asr + ocr + tabular features
Output: outputs/features/amcfg_anchor_dense_ocr_asr_features.csv
"""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # smp_vedio/
DATA_DIR = PROJECT_ROOT / "video-data"
FEATURE_DIR = PROJECT_ROOT / "outputs/features"
OUTPUT_PATH = FEATURE_DIR / "amcfg_anchor_dense_ocr_asr_features.csv"


def load_csv(path: Path, key_col: str = "pid"):
    """Load a feature CSV, deduplicate by key_col."""
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    df = df.drop_duplicates(key_col)
    return df


def main():
    print("Building AMCFG combined features...")

    # Load all parquet data for PID lists
    train_posts = pd.read_parquet(DATA_DIR / "posts_train.parquet")
    test_posts = pd.read_parquet(DATA_DIR / "posts_test.parquet")
    train_videos = pd.read_parquet(DATA_DIR / "videos_train.parquet")
    test_videos = pd.read_parquet(DATA_DIR / "videos_test.parquet")
    labels_train = pd.read_parquet(DATA_DIR / "labels_train.parquet")

    # Build base PID list
    all_posts = pd.concat([train_posts[["pid"]], test_posts[["pid"]]], ignore_index=True)
    all_videos = pd.concat([
        train_videos[["pid", "uid", "vid"]],
        test_videos[["pid", "uid", "vid"]],
    ], ignore_index=True)

    # Merge video metadata
    base = all_posts.merge(all_videos, on="pid", how="left")

    # Merge labels (popularity) for train PIDs
    train_mask = base["pid"].isin(train_posts["pid"])
    base = base.merge(labels_train[["pid", "popularity"]], on="pid", how="left")

    # Load feature CSVs
    feature_files = [
        ("video_stats", FEATURE_DIR / "video_stats.csv"),
        ("file_props", FEATURE_DIR / "video_file_props.csv"),
        ("asr", FEATURE_DIR / "video_asr.csv"),
        ("ocr", FEATURE_DIR / "frame_ocr_rapid.csv"),
        ("blip", FEATURE_DIR / "blip_video_captions_f01234567_t24.csv"),
    ]

    for name, path in feature_files:
        feat = load_csv(path)
        if feat is not None:
            # Merge on pid
            for col in feat.columns:
                if col != "pid" and col not in base.columns:
                    pid_map = dict(zip(feat["pid"], feat[col]))
                    base[col] = base["pid"].map(pid_map)
            print(f"  Merged {name}: {len(feat.columns)} columns")

    # Add basic derived features
    base["has_asr"] = (base.get("asr_char_count", pd.Series(0, index=base.index)).fillna(0) > 0).astype(int)
    base["has_ocr"] = (base.get("ocr_raw_text_count", pd.Series(0, index=base.index)).fillna(0) > 0).astype(int)
    base["has_audio"] = base.get("asr_language", pd.Series("no_audio", index=base.index)).fillna("no_audio") != "no_audio"

    # Fill NaN numeric columns with 0
    for col in base.columns:
        if col in ("pid", "uid", "vid"):
            continue
        if pd.api.types.is_numeric_dtype(base[col]):
            base[col] = base[col].fillna(0.0)

    base.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(base)} rows, {len(base.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
