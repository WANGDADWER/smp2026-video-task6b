"""
Extract ViT and CLIP embeddings from video frames.
Uses timm (ViT-base) and open_clip (ViT-L/14).
Outputs:
  outputs/features/vit_base_frame8.csv
  outputs/features/clip_vitl14_frame8_temporal.csv
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import timm
from tqdm import tqdm

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_ROOT = PROJECT_ROOT / "video_file"
DATA_DIR = PROJECT_ROOT / "video-data"
OUTPUT_DIR = PROJECT_ROOT / "outputs/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_PER_VIDEO = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIT_OUTPUT = OUTPUT_DIR / "vit_base_frame8.csv"
CLIP_OUTPUT = OUTPUT_DIR / "clip_vitl14_frame8_temporal.csv"
CACHE_PATH = OUTPUT_DIR / "embed_cache.pkl"


def build_video_list():
    rows = []
    for split in ("train", "test"):
        df = pd.read_parquet(DATA_DIR / f"videos_{split}.parquet")
        for _, row in df.iterrows():
            vp = VIDEO_ROOT / split / str(row["uid"]) / f"{row['vid']}.mp4"
            if vp.exists():
                rows.append({"pid": row["pid"], "video_path": str(vp)})
    return pd.DataFrame(rows)


def extract_frames(video_path, n_frames=FRAMES_PER_VIDEO, size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), min(n_frames, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (size, size))
            frames.append(frame)
    cap.release()
    if not frames:
        return None
    return frames


def preprocess_frames(frames):
    arr = np.stack([f.astype(np.float32) / 255.0 for f in frames])  # (N,H,W,C)
    arr = arr[..., ::-1].copy()  # BGR→RGB (copy needed for torch)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (N,C,H,W)
    return torch.from_numpy(arr)


@torch.no_grad()
def extract_vit(images, model, data_cfg):
    mean = torch.tensor(data_cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    x = images.to(DEVICE)
    x = (x - mean) / std
    feats = model.forward_features(x)
    if isinstance(feats, dict):
        feats = feats.get("pre_logits", list(feats.values())[-1])
    if feats.dim() == 3:
        feats = feats[:, 0, :]  # take CLS token: (N, tokens, D) → (N, D)
    return feats.cpu().numpy().astype(np.float32).flatten()


@torch.no_grad()
def extract_clip(images, model, processor):
    x = images.to(DEVICE)
    x = (x - 0.5) * 2.0  # rough normalize to [-1,1]
    outputs = model(pixel_values=x)
    feats = outputs.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32).flatten()


def main():
    print(f"Device: {DEVICE}")

    # Load ViT model
    print("Loading ViT-base...")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    vit = vit.to(DEVICE).eval()
    vit_cfg = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

    # Load CLIP model from local HuggingFace cache
    print("Loading CLIP ViT-L/14...")
    from transformers import CLIPVisionModel, CLIPImageProcessor
    local_clip = "/user_home/weiwenfei/.cache/clip-vit-large-patch14"
    clip_model = CLIPVisionModel.from_pretrained(local_clip).to(DEVICE).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(local_clip)

    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache: {len(cache)} videos")

    videos = build_video_list()
    print(f"Total videos: {len(videos)}")

    for _, row in tqdm(videos.iterrows(), total=len(videos), desc="Embed"):
        pid = row["pid"]
        if pid in cache:
            continue

        frames = extract_frames(row["video_path"])
        if frames is None or len(frames) == 0:
            cache[pid] = None
            continue

        images = preprocess_frames(frames)
        try:
            vit_vec = extract_vit(images, vit, vit_cfg)
            clip_vec = extract_clip(images, clip_model, clip_processor)
            cache[pid] = {"vit": vit_vec, "clip": clip_vec}
        except Exception as e:
            if len(cache) < 5:
                import traceback
                print(f"\nERROR pid={pid}: {e}")
                traceback.print_exc()
            cache[pid] = None

        if len(cache) % 100 == 0:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    # Build output CSVs
    vit_dim = None
    clip_dim = None
    for v in cache.values():
        if v is not None:
            vit_dim = len(v["vit"])
            clip_dim = len(v["clip"])
            break

    if vit_dim is None:
        print("ERROR: No embeddings extracted")
        return

    vit_rows, clip_rows = [], []
    for _, row in videos.iterrows():
        pid = row["pid"]
        v = cache.get(pid)
        vit_row = {"pid": pid}
        clip_row = {"pid": pid}
        if v is not None:
            for i in range(vit_dim):
                vit_row[f"vit_{i}"] = float(v["vit"][i])
            for i in range(clip_dim):
                clip_row[f"clip_{i}"] = float(v["clip"][i])
        else:
            for i in range(vit_dim):
                vit_row[f"vit_{i}"] = 0.0
            for i in range(clip_dim):
                clip_row[f"clip_{i}"] = 0.0
        vit_rows.append(vit_row)
        clip_rows.append(clip_row)

    pd.DataFrame(vit_rows).to_csv(VIT_OUTPUT, index=False)
    pd.DataFrame(clip_rows).to_csv(CLIP_OUTPUT, index=False)
    print(f"\nSaved {VIT_OUTPUT} ({len(vit_rows)} rows, {vit_dim} dims)")
    print(f"Saved {CLIP_OUTPUT} ({len(clip_rows)} rows, {clip_dim} dims)")


if __name__ == "__main__":
    main()
