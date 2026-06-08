#!/usr/bin/env python3
"""
Stage 2: MLP + candidate_336 Centered Residual Blend
=====================================================

Takes candidate_336 from Stage 1 and applies a centered log-residual blend
using a 5-seed MLP trained on Config C features.

Usage:
  cd stage2 && python run_train.py              # Full pipeline
  cd stage2 && python run_train.py --predict-only  # Skip CV
  cd stage2 && python run_train.py --alpha 0.05    # Custom blend alpha
"""

from __future__ import annotations

import argparse, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG / "stage2"))

from src.temporal_cv import (
    TEMPORAL_SPLITS, load_raw_data, load_visual_features,
    _make_sentences, assemble_feature_matrix,
    mape, filter_outliers,
)
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ────────────────────────────────────────────────────────────────
CANDIDATE_336 = _PKG / "stage1" / "submissions" / "candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv"
STAGE1_SUBMISSIONS = _PKG / "stage1" / "submissions"
FINAL_SUBMISSION = _PKG / "submissions" / "final_submission.csv"
OUTPUT_DIR = _PKG / "submissions"
EPS = 1e-6
SEEDS = (42, 2026, 2027, 2028, 2029)

MLP_CONFIG = {
    "hidden_dims": (256, 128, 64),
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 200,
    "patience": 30,
    "batch_size": 128,
}


# ═══════════════════════════════════════════════════════════════════════════
# MLP Architecture
# ═══════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def huber_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = diff.abs()
    return torch.where(abs_diff < delta, 0.5 * diff.pow(2),
                       delta * (abs_diff - 0.5 * delta)).mean()


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = huber_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for (xb,) in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ═══════════════════════════════════════════════════════════════════════════
# Temporal CV
# ═══════════════════════════════════════════════════════════════════════════

def run_temporal_cv(config=None):
    if config is None: config = MLP_CONFIG
    posts, users, videos, labels = load_raw_data()
    visual_features = load_visual_features()
    posts["post_time"] = pd.to_datetime(posts["post_time"])
    posts["month_key"] = posts["post_time"].dt.strftime("%Y-%m")

    posts_train_only = posts[posts["pid"].isin(labels["pid"])]
    from src.temporal_cv import W2V_PATH as _W2V
    if _W2V.exists():
        w2v = Word2Vec.load(str(_W2V))
    else:
        w2v = Word2Vec(_make_sentences(posts_train_only), vector_size=64, window=5,
                        min_count=2, workers=4, epochs=20, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    all_y_true, all_preds, split_mapes = [], [], []
    for split_name, train_months, val_months in TEMPORAL_SPLITS:
        posts_hist = posts[posts["month_key"].isin(train_months) & posts["pid"].isin(labels["pid"])].copy()
        posts_val = posts[posts["month_key"].isin(val_months) & posts["pid"].isin(labels["pid"])].copy()
        if len(posts_hist) < 100 or len(posts_val) < 50:
            print(f"  {split_name}: SKIP"); continue

        labels_hist = labels[labels["pid"].isin(posts_hist["pid"])]
        all_pids = set(posts_hist["pid"]) | set(posts_val["pid"])
        all_uids = set(posts_hist["uid"]) | set(posts_val["uid"])
        users_split = users[users["uid"].isin(all_uids)]
        videos_split = videos[videos["pid"].isin(all_pids)]

        X_hist, X_val = assemble_feature_matrix(
            posts_hist, posts_val, users_split, videos_split, labels_hist,
            w2v_model=w2v, visual_features=visual_features,
            use_videomae=True, videomae_dims=64,
            use_enhanced_tags=False, use_tfidf_svd=True, tfidf_dims=64)

        y_hist = labels_hist.set_index("pid")["popularity"]
        X_hist, y_hist = X_hist.loc[X_hist.index.intersection(y_hist.index)], y_hist.loc[X_hist.index.intersection(y_hist.index)]
        y_val = labels[labels["pid"].isin(posts_val["pid"])].set_index("pid")["popularity"]
        X_val, y_val = X_val.loc[X_val.index.intersection(y_val.index)], y_val.loc[X_val.index.intersection(y_val.index)]

        mask, lo, hi = filter_outliers(y_hist)
        X_hist_f, y_hist_f = X_hist[mask], y_hist[mask]

        numeric_cols = [c for c in X_hist_f.columns if X_hist_f[c].dtype != "object"]
        scaler = StandardScaler()
        X_hist_s = X_hist_f.copy(); X_val_s = X_val.copy()
        X_hist_s[numeric_cols] = scaler.fit_transform(X_hist_f[numeric_cols].values)
        X_val_s[numeric_cols] = scaler.transform(X_val[numeric_cols].values)

        X_tr = torch.FloatTensor(X_hist_s[numeric_cols].values.astype(np.float32))
        y_tr = torch.FloatTensor(np.log(np.clip(y_hist_f.values, EPS, None)).astype(np.float32))
        X_va = torch.FloatTensor(X_val_s[numeric_cols].values.astype(np.float32))

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va), batch_size=config["batch_size"] * 2)

        model = MLP(X_tr.shape[1], **{k: config[k] for k in ["hidden_dims","dropout"]}).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * (len(X_tr)//config["batch_size"]+1))

        best_mape, best_state, p_ct, best_ep = float("inf"), None, 0, 0
        for ep in range(config["epochs"]):
            train_one_epoch(model, loader, optimizer, scheduler, device)
            lp = predict(model, val_loader, device)
            m = mape(y_val.values, np.exp(lp))
            if m < best_mape:
                best_mape, best_state, p_ct, best_ep = m, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
            else:
                p_ct += 1
                if p_ct >= config["patience"]: break
        model.load_state_dict(best_state)
        fp = np.exp(predict(model, val_loader, device))
        split_mapes.append(mape(y_val.values, fp))
        all_y_true.extend(y_val.values.tolist()); all_preds.extend(fp.tolist())
        print(f"  {split_name}: MAPE={split_mapes[-1]:.4f} (best_epoch={best_ep})")

    g = mape(np.array(all_y_true), np.array(all_preds))
    m = np.mean(split_mapes)
    print(f"\n  CV Mean MAPE: {m:.4f}, Global OOF: {g:.4f}")
    return m, g


# ═══════════════════════════════════════════════════════════════════════════
# Final Training
# ═══════════════════════════════════════════════════════════════════════════

def train_final_mlp(config=None):
    if config is None: config = MLP_CONFIG
    posts, users, videos, labels = load_raw_data()
    visual_features = load_visual_features()
    posts["post_time"] = pd.to_datetime(posts["post_time"])
    train_pids = set(labels["pid"])
    posts_train = posts[posts["pid"].isin(train_pids)].copy()
    posts_test = posts[~posts["pid"].isin(train_pids)].copy()
    print(f"  Train: {len(posts_train)}, Test: {len(posts_test)}")

    from src.temporal_cv import W2V_PATH as _W2V
    w2v = Word2Vec.load(str(_W2V)) if _W2V.exists() else Word2Vec(
        _make_sentences(posts_train), vector_size=64, window=5, min_count=2, workers=4, epochs=20, seed=42)

    all_uids, all_pids = set(posts["uid"]), set(posts["pid"])
    X_train, X_test = assemble_feature_matrix(
        posts_train, posts_test, users[users["uid"].isin(all_uids)],
        videos[videos["pid"].isin(all_pids)], labels, w2v_model=w2v,
        visual_features=visual_features, use_videomae=True, videomae_dims=64,
        use_enhanced_tags=False, use_tfidf_svd=True, tfidf_dims=64)

    y_train = labels.set_index("pid")["popularity"]
    common = X_train.index.intersection(y_train.index)
    X_train, y_train = X_train.loc[common], y_train.loc[common]
    mask, lo, hi = filter_outliers(y_train)
    X_train_f, y_train_f = X_train[mask], y_train[mask]
    print(f"  After filter: {len(y_train_f)} (bounds [{lo:.1f}, {hi:.1f}])")

    numeric_cols = [c for c in X_train_f.columns if X_train_f[c].dtype != "object"]
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train_f.copy(), X_test.copy()
    X_train_s[numeric_cols] = scaler.fit_transform(X_train_f[numeric_cols].values)
    X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols].values)

    y_log = np.log(np.clip(y_train_f.values, EPS, None)).astype(np.float32)
    X_tr = torch.FloatTensor(X_train_s[numeric_cols].values.astype(np.float32))
    X_te = torch.FloatTensor(X_test_s[numeric_cols].values.astype(np.float32))
    y_tr = torch.FloatTensor(y_log)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train, ds_test = TensorDataset(X_tr, y_tr), TensorDataset(X_te)
    preds_list = []

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        loader = DataLoader(ds_train, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(ds_test, batch_size=config["batch_size"] * 2)
        model = MLP(X_tr.shape[1], **{k: config[k] for k in ["hidden_dims","dropout"]}).to(device)
        opt = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"]*(len(ds_train)//config["batch_size"]+1))
        for _ in range(config["epochs"]):
            train_one_epoch(model, loader, opt, sched, device)
        p = np.clip(np.exp(predict(model, test_loader, device)), 1.0, None)
        preds_list.append(p)
        print(f"  seed={seed}: mean={p.mean():.4f}, std={p.std():.4f}")

    log_preds = [np.log(np.clip(p, EPS, None)) for p in preds_list]
    final_pred = np.exp(np.mean(log_preds, axis=0))
    return pd.DataFrame({"pid": X_test_s.index, "polularity_score": final_pred}), preds_list


# ═══════════════════════════════════════════════════════════════════════════
# Blending
# ═══════════════════════════════════════════════════════════════════════════

def generate_blends(mlp_sub, alphas=(0.01, 0.02, 0.03, 0.05, 0.08)):
    pub = pd.read_csv(CANDIDATE_336).set_index("pid")["polularity_score"].astype(float)
    mlp = mlp_sub.set_index("pid")["polularity_score"].astype(float).loc[pub.index]
    log_pub = np.log(np.clip(pub.values.astype(np.float64), EPS, None))
    log_mlp = np.log(np.clip(mlp.values.astype(np.float64), EPS, None))
    r = log_mlp - log_pub; r = r - r.mean()
    results = {}
    for a in alphas:
        pred = np.clip(pub.values * np.exp(a * r), 1.0, None)
        results[a] = pd.DataFrame({"pid": pub.index, "polularity_score": pred})
        print(f"  α={a:.3f}: mean={pred.mean():.4f}, std={pred.std():.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 2: MLP + candidate_336 Blend")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    config = dict(MLP_CONFIG); config["epochs"] = args.epochs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 2: MLP [256,128,64] + candidate_336 Centered Blend")
    print(f"Candidate: {CANDIDATE_336}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)

    if not args.predict_only:
        print("\n[Step 1] Temporal CV...")
        run_temporal_cv(config)

    if not args.cv_only:
        print(f"\n[Step 2] Final MLP Training ({len(SEEDS)} seeds)...")
        mlp_sub, preds_list = train_final_mlp(config)
        path_base = OUTPUT_DIR / "stage2_mlp_base.csv"
        mlp_sub.to_csv(path_base, index=False)
        print(f"  Saved: {path_base}")

        for seed, p in zip(SEEDS, preds_list):
            pd.DataFrame({"pid": mlp_sub["pid"], "polularity_score": p}).to_csv(
                OUTPUT_DIR / f"stage2_mlp_seed{seed}.csv", index=False)

        print(f"\n[Step 3] Blends with candidate_336...")
        blends = generate_blends(mlp_sub, sorted(set([args.alpha, 0.01, 0.02, 0.03, 0.05, 0.08])))
        for a, sub in blends.items():
            p = OUTPUT_DIR / f"stage2_centered_a{a:.3f}.csv"
            sub.to_csv(p, index=False)

        # Copy best to final_submission.csv
        best = OUTPUT_DIR / f"stage2_centered_a{args.alpha:.3f}.csv"
        if best.exists():
            shutil.copy(best, FINAL_SUBMISSION)
            print(f"\n  Final submission: {FINAL_SUBMISSION}")

    print("Done.")


if __name__ == "__main__":
    main()
