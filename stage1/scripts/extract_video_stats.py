"""
Extract video statistics from video files.
Output:
  outputs/features/video_stats.csv {pid, duration, fps, width, height, aspect_ratio, total_frames}
  outputs/features/video_file_props.csv {pid, file_size_bytes, file_exists}
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_ROOT = PROJECT_ROOT / "video_file"
DATA_DIR = PROJECT_ROOT / "video-data"
OUTPUT_DIR = PROJECT_ROOT / "outputs/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_video_list():
    rows = []
    for split in ("train", "test"):
        parquet_path = DATA_DIR / f"videos_{split}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            video_path = VIDEO_ROOT / split / str(row["uid"]) / f"{row['vid']}.mp4"
            rows.append({
                "pid": row["pid"],
                "video_path": str(video_path),
                "file_exists": video_path.exists(),
                "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
            })
    return pd.DataFrame(rows)


def main():
    videos = build_video_list()
    print(f"Total videos: {len(videos)}")
    print(f"Files exist: {videos['file_exists'].sum()}")

    stats_rows = []
    for _, row in tqdm(videos.iterrows(), total=len(videos), desc="Video stats"):
        pid = row["pid"]
        if not row["file_exists"]:
            stats_rows.append({
                "pid": pid, "duration": np.nan, "fps": np.nan,
                "width": np.nan, "height": np.nan,
                "aspect_ratio": np.nan, "total_frames": 0,
            })
            continue

        cap = cv2.VideoCapture(row["video_path"])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else np.nan
        aspect = width / height if height > 0 else np.nan
        cap.release()

        stats_rows.append({
            "pid": pid,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "total_frames": total_frames,
        })

    stats_df = pd.DataFrame(stats_rows)
    props_df = videos[["pid", "file_size_bytes", "file_exists"]].copy()

    stats_path = OUTPUT_DIR / "video_stats.csv"
    props_path = OUTPUT_DIR / "video_file_props.csv"
    stats_df.to_csv(stats_path, index=False)
    props_df.to_csv(props_path, index=False)

    print(f"\nSaved: {stats_path} ({len(stats_df)} rows, {len(stats_df.columns)} cols)")
    print(f"Saved: {props_path} ({len(props_df)} rows)")

    # Summary
    valid = stats_df["duration"].notna()
    print(f"Videos with valid stats: {valid.sum()}/{len(stats_df)}")
    if valid.sum() > 0:
        print(f"Duration: mean={stats_df.loc[valid,'duration'].mean():.1f}s, "
              f"median={stats_df.loc[valid,'duration'].median():.1f}s")
        print(f"Resolution: most common "
              f"{stats_df.loc[valid,'width'].mode().iloc[0]:.0f}x"
              f"{stats_df.loc[valid,'height'].mode().iloc[0]:.0f}")


if __name__ == "__main__":
    main()
