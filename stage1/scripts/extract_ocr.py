"""
Extract OCR text from video frames using RapidOCR (ONNX runtime).
Processes all train+test videos.
Output: outputs/features/frame_ocr_rapid.csv {pid, ocr_text, ocr_raw_text_count}
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # smp_vedio/
VIDEO_ROOT = PROJECT_ROOT / "video_file"
DATA_DIR = PROJECT_ROOT / "video-data"
OUTPUT_DIR = PROJECT_ROOT / "outputs/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "frame_ocr_rapid.csv"
CACHE_PATH = OUTPUT_DIR / "frame_ocr_cache.pkl"
FRAMES_PER_VIDEO = 4


def build_video_list():
    rows = []
    for split in ("train", "test"):
        parquet_path = DATA_DIR / f"videos_{split}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            video_path = VIDEO_ROOT / split / str(row["uid"]) / f"{row['vid']}.mp4"
            if video_path.exists():
                rows.append({"pid": row["pid"], "video_path": str(video_path)})
    return pd.DataFrame(rows)


def extract_frames(video_path: str, n_frames: int = FRAMES_PER_VIDEO) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    frames = []
    indices = np.linspace(0, max(total_frames - 1, 0), min(n_frames, total_frames), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def main():
    engine = RapidOCR()
    print("RapidOCR engine initialized")

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache: {len(cache)} videos already processed")

    videos = build_video_list()
    print(f"Total videos to process: {len(videos)}")

    for _, row in tqdm(videos.iterrows(), total=len(videos), desc="OCR"):
        pid = row["pid"]
        if pid in cache:
            continue

        frames = extract_frames(row["video_path"], FRAMES_PER_VIDEO)
        if not frames:
            cache[pid] = {"ocr_text": "", "ocr_raw_text_count": 0}
            continue

        all_texts = []
        for frame in frames:
            try:
                result, _ = engine(frame)
                if result:
                    for (_box, text, _conf) in result:
                        if text and str(text).strip():
                            all_texts.append(str(text).strip())
            except Exception:
                pass

        ocr_text = " | ".join(all_texts) if all_texts else ""
        cache[pid] = {"ocr_text": ocr_text, "ocr_raw_text_count": len(ocr_text)}

        if len(cache) % 100 == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    out_rows = [{"pid": row["pid"], **cache.get(row["pid"], {"ocr_text": "", "ocr_raw_text_count": 0})}
                for _, row in videos.iterrows()]
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    n_text = (out_df["ocr_raw_text_count"] > 0).sum()
    print(f"\nSaved {len(out_df)} rows to {OUTPUT_PATH}")
    print(f"With OCR text: {n_text} videos")


if __name__ == "__main__":
    main()
