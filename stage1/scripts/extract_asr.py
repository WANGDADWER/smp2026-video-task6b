"""
Extract ASR (Automatic Speech Recognition) text from video audio.
Uses OpenAI Whisper base model + moviepy for audio extraction.
Processes all train+test videos.
Output: outputs/features/video_asr.csv {pid, asr_text, asr_language, asr_char_count}
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
import whisper
from moviepy import VideoFileClip
from tqdm import tqdm

# Config — resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # smp_vedio/
VIDEO_ROOT = PROJECT_ROOT / "video_file"
DATA_DIR = PROJECT_ROOT / "video-data"
OUTPUT_DIR = PROJECT_ROOT / "outputs/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "video_asr.csv"
CACHE_PATH = OUTPUT_DIR / "video_asr_cache.pkl"
AUDIO_DURATION = 30  # seconds of audio to extract per video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_video_list():
    """Return DataFrame with pid, uid, vid, split for all videos."""
    rows = []
    for split in ("train", "test"):
        parquet_path = DATA_DIR / f"videos_{split}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            video_path = VIDEO_ROOT / split / str(row["uid"]) / f"{row['vid']}.mp4"
            if video_path.exists():
                rows.append({
                    "pid": row["pid"],
                    "uid": row["uid"],
                    "vid": row["vid"],
                    "video_path": str(video_path),
                })
    return pd.DataFrame(rows)


def extract_audio(video_path: str, max_duration: int = AUDIO_DURATION) -> np.ndarray | None:
    """Extract audio from video using moviepy, return float32 numpy array at 16kHz mono."""
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        duration = min(clip.audio.duration, max_duration)
        if duration < 0.5:
            clip.close()
            return None
        # Extract subclip and get audio as numpy array
        audio_clip = clip.audio.subclipped(0, duration)
        source_sr = int(clip.audio.fps)  # must read before close()
        arr = audio_clip.to_soundarray()  # shape: (nsamples, nchannels)
        audio_clip.close()
        clip.close()
        # Convert to mono by averaging channels
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.mean(axis=1)
        elif arr.ndim == 2:
            arr = arr[:, 0]
        arr = arr.astype(np.float32)
        # Resample to 16kHz if needed (moviepy typically uses 44100)
        target_sr = 16000
        if source_sr != target_sr and len(arr) > 0:
            import scipy.signal
            num_samples = int(len(arr) * target_sr / source_sr)
            arr = scipy.signal.resample(arr, num_samples).astype(np.float32)
        return arr
    except Exception:
        return None


def main():
    print(f"Device: {DEVICE}")
    model = whisper.load_model("base", device=DEVICE)

    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache: {len(cache)} videos already processed")

    videos = build_video_list()
    print(f"Total videos to process: {len(videos)}")

    for _, row in tqdm(videos.iterrows(), total=len(videos), desc="ASR"):
        pid = row["pid"]
        if pid in cache:
            continue

        video_path = row["video_path"]

        # Extract audio
        audio = extract_audio(video_path, AUDIO_DURATION)
        if audio is None or len(audio) < 8000:  # less than 0.5 seconds at 16kHz
            entry = {"asr_text": "", "asr_language": "no_audio", "asr_char_count": 0}
            cache[pid] = entry
            continue

        # Transcribe
        try:
            result = model.transcribe(audio, fp16=(DEVICE == "cuda"), language=None)
            text = result["text"].strip()
            lang = result.get("language", "unknown")
        except Exception:
            text = ""
            lang = "error"

        entry = {
            "asr_text": text,
            "asr_language": lang,
            "asr_char_count": len(text),
        }
        cache[pid] = entry

        # Save cache periodically
        if len(cache) % 100 == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)

    # Final save
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    # Build output DataFrame
    out_rows = []
    for _, row in videos.iterrows():
        pid = row["pid"]
        entry = cache.get(pid, {"asr_text": "", "asr_language": "", "asr_char_count": 0})
        out_rows.append({"pid": pid, **entry})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    n_text = (out_df["asr_char_count"] > 0).sum()
    print(f"\nSaved {len(out_df)} rows to {OUTPUT_PATH}")
    print(f"With ASR text: {n_text} videos")
    if n_text > 0:
        print(f"Languages: {out_df['asr_language'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
