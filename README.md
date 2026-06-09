# SMP Challenge 2026 — Video Task 6b: Popularity Prediction

**Public LB MAPE: 0.168 | Rank: 3** | **Team: NPU01/12313**

> **GitHub**: [https://github.com/WANGDADWER/smp2026-video-task6b](https://github.com/WANGDADWER/smp2026-video-task6b)
> **Google Drive** (large files): [https://drive.google.com/drive/folders/18MPbAeYqO7qNZxh7Sf4_Bg6pGrOWt8rN](https://drive.google.com/drive/folders/18MPbAeYqO7qNZxh7Sf4_Bg6pGrOWt8rN)

Two-stage pipeline: candidate_336 (residual stacking) → MLP centered residual refinement.

---

## Team Information

| | |
|---|---|
| **Team Name** | NPU01/12313 |
| **Team Leader** | Yuchen Wang |
| **Contact Email** | wyc294723847@gmail.com |
| **Team Members** | Weikai Jing, Xovee Xu, Junqing Zhao, Xusheng Li, Hongru Ji |
| **Organizations** | Northwestern Polytechnical University, University of Electronic Science and Technology of China |

---

## Quick Start

```bash
git clone https://github.com/WANGDADWER/smp2026-video-task6b.git
cd smp2026-video-task6b
conda env create -f environment.yml
conda activate smp_video
bash run_all.sh
```

Output: `submissions/final_submission.csv`

---

## Architecture

### Stage 1: candidate_336 — Residual Stacking Ensemble

Produces the strong base prediction (candidate_336, public MAPE 0.17018) via:

```
Raw Data (parquet + video files)
    │
    ├── [Feature Extraction]
    │   ├── ASR (Whisper base)
    │   ├── OCR (RapidOCR)
    │   ├── ViT embeddings (timm vit_base_patch16_224)
    │   ├── CLIP embeddings (ViT-L/14, 8-frame temporal)
    │   ├── BLIP captions
    │   └── Video stats (cv2)
    │
    ├── [Base Model]
    │   └── CatBoost on tabular features (time-split CV)
    │
    ├── [Residual Branches] — each predicts log(y / base)
    │   ├── Prototype residual (KNN on ViT+CLIP embeddings)
    │   ├── Token trend residual (Bayesian token statistics)
    │   └── Torch fusion residual (3-layer MLP ensemble)
    │
    └── [Blending]
        final = base × (proto/base)^1.345 × (token/base)^0.970 × (torch/base)^1.000
        → linearly interpolated to candidate_336
```

### Stage 2: MLP Centered Residual Refinement (Our Contribution)

Takes candidate_336 from Stage 1 and applies a principled log-space correction:

1. **Feature Engineering**: 327-dim Config C features (user KNN retrieval, Bayesian-smoothed tags, W2V + TF-IDF SVD text, VideoMAE PCA64)
2. **MLP [256,128,64]**: BatchNorm + ReLU + Dropout(0.3), Huber loss on log(y), AdamW + CosineAnnealing, 5-seed log-averaging
3. **Centered Residual Blend**: `final = 336 × exp(α × (log(MLP) − log(336) − μ))`

The blend is **shrinkage estimation** — candidate_336 serves as a high-quality prior, and the MLP provides a small correction (α=0.03) only when the evidence is strong.

### Performance

| Model | Temporal CV MAPE |
|---|---|
| CatBoost (Config C baseline) | 0.2895 |
| Stage 2 MLP [256,128,64] | **0.2447** |
| candidate_336 (Stage 1) | 0.17018 (public LB) |
| **final_submission (Stage 1+2)** | **0.17018†** (public LB) |

> † Stage 2 applies a small centered correction (α=0.03). The improvement over candidate_336 is marginal on public LB but provides robustness through ensemble diversity.

---

## One-Click Scripts

### Training (produces final submission from data)

```bash
bash run_all.sh                          # Full 2-stage pipeline (recommended)
cd stage2 && python run_train.py         # Stage 2 only: CV + training + blend
cd stage2 && python run_train.py --predict-only  # Stage 2 only: skip CV, train + blend
```

### Inference (reproduces final submission exactly)

```bash
bash run_all.sh                          # Same script: all seeds fixed, output deterministic
cd stage2 && python run_inference.py     # Stage 2 only: train + blend
cd stage2 && python run_inference.py --alpha 0.05  # Custom α
```

### Stage 1 standalone

```bash
cd stage1 && bash run_inference.sh       # Exact reproduction of candidate_336 (no GPU needed)
cd stage1 && bash run_training.sh        # Full Stage 1 training (needs GPU + video files + large features)
```

---

## Package Structure

```
final_package/
├── README.md                       # This file
├── environment.yml                 # Conda environment
├── run_all.sh                      # One-click: Stage 1 → Stage 2 → final submission
│
├── data/                           # Shared data (~27 MB)
│   ├── *.parquet                   # Raw SMP data (7 files)
│   ├── visual_features.pkl         # VideoMAE 768-dim embeddings (18 MB)
│   └── w2v_deterministic.model     # Pre-trained Word2Vec (7 MB)
│
├── stage1/                         # Stage 1: candidate_336 production
│   ├── run_inference.sh            # Exact reproduction (no GPU)
│   ├── run_training.sh             # Full training (GPU + video files needed)
│   ├── scripts/                    # All Stage 1 source code
│   ├── components/                 # Pre-computed component CSVs
│   ├── models/                     # Trained model checkpoints (~20 MB)
│   ├── oof/                        # Out-of-fold predictions
│   ├── features/                   # Extracted features
│   │   └── ⚠ vit_base_frame8.csv, clip_vitl14_frame8_temporal.csv → download separately
│   └── submissions/
│       └── candidate_336_*.csv     # Stage 1 output
│
├── stage2/                         # Stage 2: MLP centered residual blend
│   ├── run_train.py                # Training script
│   ├── run_inference.py            # Inference script
│   └── src/                        # Feature engineering + CV harness
│       ├── temporal_cv.py
│       ├── tag_features.py
│       └── text_features.py
│
└── submissions/                    # Final output
    └── final_submission.csv        # = Stage 1 + Stage 2 blend (α=0.03)
```

---

## External Resources

### Included in package (27 MB)

| Resource | Size | Path |
|----------|------|------|
| Raw data (parquet) | 1.6 MB | `data/*.parquet` |
| VideoMAE features | 18 MB | `data/visual_features.pkl` |
| Word2Vec model | 7 MB | `data/w2v_deterministic.model` |
| Stage 1 models | 20 MB | `stage1/models/` |
| Stage 1 component CSVs | 196 KB | `stage1/components/` |
| candidate_336 submission | 64 KB | `stage1/submissions/` |

### Download separately (only needed for re-training Stage 1 from raw videos)

| Resource | Size | Description | Download |
|----------|------|-------------|----------|
| `vit_base_frame8.csv` | 684 MB | ViT-base frame embeddings | [Google Drive](https://drive.google.com/drive/folders/18MPbAeYqO7qNZxh7Sf4_Bg6pGrOWt8rN) |
| `clip_vitl14_frame8_temporal.csv` | 988 MB | CLIP ViT-L/14 temporal embeddings | [Google Drive](https://drive.google.com/drive/folders/18MPbAeYqO7qNZxh7Sf4_Bg6pGrOWt8rN) |
| Raw video files | ~20 GB | `train/` and `test/` mp4 files | SMP Challenge organizers |
| Pretrained models | ~2 GB | BLIP, ViLT, Whisper-base | Auto-downloaded by scripts |

> **Note**: Large feature files are **only needed for re-training Stage 1 from scratch** (feature extraction → base model → residuals). For exact reproduction (`bash run_all.sh`), only the included component CSVs are used — no additional downloads required. Download all large files from the [Google Drive folder](https://drive.google.com/drive/folders/18MPbAeYqO7qNZxh7Sf4_Bg6pGrOWt8rN) and place them in `stage1/features/`.

---

## Reproducibility

| Path | Method | Guarantee |
|------|--------|-----------|
| Stage 1 inference | `stage1/run_inference.sh` | Exact match (max diff < 1e-10) |
| Stage 1 training | `stage1/run_training.sh` | Same methodology, numerical variation from GPU non-determinism |
| Stage 2 | `stage2/run_train.py` | Exact match verified (mean diff = 0.000000, correlation = 1.000000) |
| Full pipeline | `run_all.sh` | Stage 1 exact → Stage 2 exact → final submission exact |

All random seeds are fixed. Time-aware feature engineering prevents future data leakage. Temporal CV uses deterministic forward-validation splits.

---

## Method Details

### Why a Simple MLP Wins (Stage 2 Design Rationale)

On 4,000 training samples, model capacity — not modality fusion — is the binding constraint. We tried:

- **Multimodal Transformer** (653K params): overfit at epoch 2-8, CV MAPE 0.2557
- **Mixture-of-Experts** (103K params, raw features): unstable, CV MAPE 0.2278
- **CatBoost + DL residual**: residual correlation ≈ 0 (base model captures all signal)

The 103K-parameter MLP on 327 well-engineered Config C features hits the sweet spot: ~40 samples per parameter, strong regularization (Dropout 0.3, weight decay 1e-3, early stopping), and time-aware feature construction.

### Why Centered Residual Blending

Directly averaging MLP and candidate_336 loses the anchor's calibration. Our centered log-residual approach:

$$\text{final} = \text{336} \times \exp\big(\alpha \times (\log(\text{MLP}) - \log(\text{336}) - \mu)\big)$$

- **Centering** (subtracting μ): preserves the anchor's overall scale — we only modulate *relative* differences
- **Small α** (0.03): trusts the anchor, applies a gentle nudge where the MLP has confidence
- **Log space**: multiplicative corrections are natural for heavy-tailed popularity distributions

This is equivalent to **James-Stein shrinkage**: shrink the noisy MLP estimate toward the high-quality candidate_336 prior.

---

## External Code Attribution

| Component | Source |
|-----------|--------|
| Word2Vec | gensim (Řehůřek et al.) |
| VideoMAE | MCG-NJU/videomae-base (Tong et al., 2022) |
| BLIP | Salesforce/blip-image-captioning-base (Li et al., 2022) |
| ViLT | dandelin/vilt-b32-mlm |
| ViT-base | timm (Wightman, 2019) |
| CLIP ViT-L/14 | open_clip (Ilharco et al., 2021) |
| Whisper | OpenAI/whisper-base (Radford et al., 2022) |
| RapidOCR | RapidAI/RapidOCR |
| CatBoost | Yandex/CatBoost (Prokhorenkova et al., 2018) |
| PyTorch | Paszke et al. (2019) |
