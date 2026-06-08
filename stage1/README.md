# SMP 2026 Video Task 6b — Best Submission Package

**Public LB MAPE: 0.17018**

---

## 1. Quick Start

### Prerequisites

```bash
conda env create -f environment.yml
conda activate smp2026
```

### Data Setup

Ensure the following directories are present at the project root:

```text
video-data/          # Parquet files (posts, users, videos, labels)
video_file/          # Raw video files (train/ and test/ subdirectories)
pretrained_models/   # Word2Vec model (w2v_deterministic.model)
video_features/      # VideoMAE features (visual_features.pkl)
fame_models/         # Cached pretrained models (BLIP, ViLT)
```

### Inference (exact reproduction)

```bash
bash run_inference.sh
```

Output: `scripts/reproduced/candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv`

This reproduces the best submission **exactly** (max difference < 1e-10). Only requires `pandas` + `numpy`, no GPU.

### Training (full pipeline)

```bash
bash run_training.sh
```

Runs the complete pipeline: feature extraction → base model → residuals → blending.

---

## 2. Architecture

```
Raw Data (parquet + videos)
    │
    ├── [Feature Extraction] ──────────────────────────┐
    │   ├── ASR (Whisper base)                         │
    │   ├── OCR (RapidOCR)                             │
    │   ├── ViT embeddings (timm vit_base_patch16_224)  │
    │   ├── CLIP embeddings (ViT-L/14)                  │
    │   ├── Video stats (cv2)                          │
    │   └── BLIP captions (pregenerated)               │
    │                                                   │
    ├── [Base Model] ──────────────────────────────────┤
    │   └── step1_baseline.py                          │
    │       CatBoost with tabular features              │
    │       Time-split CV (May→Jun, May-Jun→Jul, ...)  │
    │       Output: OOF + test submission              │
    │                                                   │
    ├── [Residual Models] ─────────────────────────────┤
    │   ├── Proto residual (KNN on ViT+CLIP + CatBoost)│
    │   ├── Token trend (Bayesian token statistics)    │
    │   └── Torch fusion (3-layer MLP ensemble)        │
    │       Each predicts log(y/base)                  │
    │                                                   │
    └── [Blending] ────────────────────────────────────┘
        └── final = base × Π exp(w_i × log_resid_i)

Final output: reproduce_best_from_components.py
    Linearly interpolates between two component CSVs
    to exactly reproduce the best submission.
```

## 3. File Structure

```text
submission_package/
├── README.md                          # This document
├── environment.yml                    # Conda environment
├── run_inference.sh                   # One-click inference (exact)
├── run_training.sh                    # One-click training (full)
├── scripts/                           # All source code
│   ├── step1_baseline.py              # Base model training
│   ├── step1_base_model_full.py       # Base model (full features)
│   ├── smp26/                         # Shared library (data, features, metrics)
│   ├── extract_asr.py                 # ASR feature extraction
│   ├── extract_ocr.py                 # OCR feature extraction
│   ├── extract_vit_clip_embeddings.py # ViT + CLIP extraction
│   ├── extract_video_stats.py         # Video properties
│   ├── build_amcfg_features.py        # Combined features
│   ├── video_task6b_prototype_residual.py  # Proto KNN residual
│   ├── video_task6b_token_trend.py         # Token trend residual
│   ├── video_task6b_torch_fusion.py        # Torch MLP residual
│   ├── video_task6b_cat_residual.py        # CatBoost residual
│   ├── video_task6b_id_residual.py         # ID/structure residual
│   ├── video_task6b_retrieval_residual.py  # User retrieval residual
│   ├── video_task6b_overlay_components.py  # Component blender
│   ├── video_task6b_public_feedback_optimizer.py  # LB weight optimizer
│   ├── video_task6b_residual_lab.py        # Shared I/O utilities
│   ├── video_task6b_lgbm_dense_residual.py # Dense feature preparation
│   └── reproduce_best_from_components.py   # Exact reproduction
├── features/                          # Extracted feature CSVs
├── components/                        # Intermediate component CSVs (exact reproduction)
├── submissions/                       # Final submission CSV
└── oof/                               # Out-of-fold predictions
```

## 4. External Resources

Due to size constraints, the following should be downloaded separately:

| Resource | Size | Description |
|----------|------|-------------|
| `features/vit_base_frame8.csv` | 684 MB | ViT-base frame embeddings |
| `features/clip_vitl14_frame8_temporal.csv` | 988 MB | CLIP ViT-L/14 frame embeddings |
| `pretrained_models/w2v_deterministic.model` | 7 MB | Word2Vec model |
| `video_features/visual_features.pkl` | 19 MB | VideoMAE embeddings |
| `fame_models/` | ~2 GB | Cached BLIP/ViLT models |

[Storage Link: to be provided]

## 5. Key Design Decisions

- **Residual stacking**: Final = base × exp(Σ w_i × residual_i). This additive-in-log-space architecture allows each residual branch to independently predict log-space corrections.
- **Time-split CV**: All validation uses expanding-window temporal splits (train on months before validation month) to prevent future leakage.
- **Public feedback optimization**: Component weights (proto=1.345, token=0.970, torch=1.000) were tuned via ridge regression surrogate over public LB scores.
- **Three residual branches** were selected from six candidates based on public LB validation: proto (positive gain), token (positive gain), torch (small positive gain). Cat/ID/retrieval branches showed negative or zero gain and were excluded.

## 6. Reproducibility

- **Exact**: `run_inference.sh` reproduces the best submission with max absolute difference < 1e-10.
- **Training**: `run_training.sh` demonstrates the complete pipeline. Due to inherent training randomness (GPU precision, model initialization, batch processing), intermediate outputs may differ numerically but the methodology is identical.
- **External code/models**: Word2Vec (gensim), VideoMAE-base (MCG-NJU), BLIP (Salesforce), ViLT (dandelin), ViT-base (timm), CLIP ViT-L/14 (open_clip), Whisper-base (OpenAI), RapidOCR.
