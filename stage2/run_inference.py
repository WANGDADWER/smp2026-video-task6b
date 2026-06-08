#!/usr/bin/env python3
"""
Stage 2: Inference Script — MLP prediction + candidate_336 blend.

Usage:
  cd stage2 && python run_inference.py
  cd stage2 && python run_inference.py --alpha 0.05
"""

from __future__ import annotations
import argparse, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG / "stage2"))

from src.temporal_cv import (load_raw_data, load_visual_features, _make_sentences,
                              assemble_feature_matrix, filter_outliers)
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CANDIDATE_336 = _PKG / "stage1" / "submissions" / "candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv"
OUTPUT_DIR = _PKG / "submissions"
FINAL_SUBMISSION = OUTPUT_DIR / "final_submission.csv"
EPS, SEEDS = 1e-6, (42, 2026, 2027, 2028, 2029)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); preds = []
    for (xb,) in loader: preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Inference")
    parser.add_argument("--alpha", type=float, default=0.03)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    posts, users, videos, labels = load_raw_data()
    visual_features = load_visual_features()
    posts["post_time"] = pd.to_datetime(posts["post_time"])
    train_pids = set(labels["pid"])
    posts_train = posts[posts["pid"].isin(train_pids)].copy()
    posts_test = posts[~posts["pid"].isin(train_pids)].copy()

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

    numeric_cols = [c for c in X_train_f.columns if X_train_f[c].dtype != "object"]
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train_f.copy(), X_test.copy()
    X_train_s[numeric_cols] = scaler.fit_transform(X_train_f[numeric_cols].values)
    X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols].values)

    y_log = np.log(np.clip(y_train_f.values, EPS, None)).astype(np.float32)
    X_tr = torch.FloatTensor(X_train_s[numeric_cols].values.astype(np.float32))
    X_te = torch.FloatTensor(X_test_s[numeric_cols].values.astype(np.float32))
    y_tr = torch.FloatTensor(y_log)

    ds_train, ds_test = TensorDataset(X_tr, y_tr), TensorDataset(X_te)
    preds_list = []

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        loader = DataLoader(ds_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(ds_test, batch_size=256)
        model = MLP(X_tr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200*(len(ds_train)//128+1))
        for _ in range(200):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device); opt.zero_grad()
                diff = model(xb) - yb
                loss = torch.where(diff.abs()<1.0, 0.5*diff.pow(2), (diff.abs()-0.5)).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        p = np.clip(np.exp(predict(model, test_loader, device)), 1.0, None)
        preds_list.append(p)
        print(f"  seed={seed}: mean={p.mean():.4f}, std={p.std():.4f}")

    mlp_pred = np.exp(np.mean([np.log(np.clip(p, EPS, None)) for p in preds_list], axis=0))
    mlp_sub = pd.DataFrame({"pid": X_test.index, "polularity_score": mlp_pred})
    path = OUTPUT_DIR / "stage2_mlp_base.csv"; mlp_sub.to_csv(path, index=False)
    print(f"MLP saved: {path}")

    pub = pd.read_csv(CANDIDATE_336).set_index("pid")["polularity_score"].astype(float)
    mlp = mlp_sub.set_index("pid")["polularity_score"].astype(float).loc[pub.index]
    log_pub = np.log(np.clip(pub.values, EPS, None))
    log_mlp = np.log(np.clip(mlp.values, EPS, None))
    r = log_mlp - log_pub; r = r - r.mean()
    final = np.clip(pub.values * np.exp(args.alpha * r), 1.0, None)
    sub = pd.DataFrame({"pid": pub.index, "polularity_score": final})
    path = OUTPUT_DIR / f"stage2_centered_a{args.alpha:.3f}.csv"; sub.to_csv(path, index=False)
    shutil.copy(path, FINAL_SUBMISSION)
    print(f"Final: {FINAL_SUBMISSION} (α={args.alpha:.3f}, mean={final.mean():.4f})")


if __name__ == "__main__":
    main()
