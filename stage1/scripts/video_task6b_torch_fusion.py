from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape
from video_task6b_cat_residual import add_features, attach_feature_csvs
from video_task6b_lgbm_dense_residual import add_basic_dense_features
from video_task6b_prototype_residual import embedding_columns, parse_float_list, safe_name
from video_task6b_residual_lab import aligned_values, read_submission, write_submission


SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embedding_pca(path: str, train_pids: pd.Series, test_pids: pd.Series, n_components: int, seed: int):
    usecols = embedding_columns(path)
    feat = pd.read_csv(path, usecols=usecols).drop_duplicates("pid")
    pids = pd.concat([train_pids, test_pids], ignore_index=True).to_frame("pid")
    x = pids.merge(feat, on="pid", how="left")
    cols = [c for c in x.columns if c != "pid"]
    arr = x[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = np.where(~np.isfinite(arr))
    if len(bad[0]):
        arr[bad] = med[bad[1]]
    scaler = StandardScaler()
    arr = scaler.fit_transform(arr).astype(np.float32)
    dims = min(n_components, arr.shape[1], arr.shape[0] - 1)
    pca = PCA(n_components=dims, random_state=seed, svd_solver="randomized")
    arr = pca.fit_transform(arr).astype(np.float32)
    prefix = safe_name(path).replace("-", "_")
    out = pd.DataFrame(arr, columns=[f"{prefix}_pca_{i:03d}" for i in range(arr.shape[1])])
    out.insert(0, "pid", pids["pid"].to_numpy())
    return out.iloc[: len(train_pids)].reset_index(drop=True), out.iloc[len(train_pids) :].reset_index(drop=True), {
        "path": path,
        "raw_dims": len(cols),
        "pca_dims": int(arr.shape[1]),
        "explained": float(np.sum(pca.explained_variance_ratio_)),
    }


def prepare_frames(args):
    train, test = load_train_test(args.data_dir)
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)
    train, test = attach_feature_csvs(train, test, args.feature_csvs)
    train, test = add_features(train, test)
    train, test = add_basic_dense_features(train, test)
    train = train.drop_duplicates("pid")
    test = test.drop_duplicates("pid")
    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    test["month_key"] = test["dt"].dt.strftime("%Y-%m")
    oof = pd.read_csv(args.oof)
    train = train.merge(oof[["pid", args.oof_pred_col]].rename(columns={args.oof_pred_col: "oof_pred"}), on="pid", how="inner")
    train["oof_pred"] = np.clip(pd.to_numeric(train["oof_pred"], errors="coerce"), 1e-6, None)
    emb_meta = []
    for path in args.embedding_csvs:
        tr, te, meta = load_embedding_pca(path, train["pid"], test["pid"], args.pca_components, args.seed)
        emb_meta.append(meta)
        train = pd.concat([train.reset_index(drop=True), tr.drop(columns=["pid"])], axis=1)
        test = pd.concat([test.reset_index(drop=True), te.drop(columns=["pid"])], axis=1)
    return train, test, emb_meta


def numeric_cols(df: pd.DataFrame):
    skip = {
        "pid",
        "uid",
        "vid",
        "post_time",
        "video_path",
        "post_content",
        "post_suggested_words",
        "dt",
        "split",
        "popularity",
        "oof_pred",
        "month_key",
        "content_text",
        "suggested_text",
        "music_text",
        "full_text",
        "asr_text",
        "ocr_text",
        "blip_caption",
    }
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def make_matrix(train_ref: pd.DataFrame, df: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None):
    ref = train_ref[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = ref.median().fillna(0.0)
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)
    else:
        x = scaler.transform(x).astype(np.float32)
    return x, scaler


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_predict(x_train, y_train, x_target, args, seed):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MLP(x_train.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.08)
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model.train()
    for epoch in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_target), 1024):
            xb = torch.tensor(x_target[start : start + 1024], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds)


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vv = v - float(np.mean(v))
    b = basis - float(np.mean(basis))
    denom = float(np.dot(b, b))
    if denom <= 1e-12:
        return v
    beta = float(np.dot(vv, b) / denom)
    return v - beta * basis


def main():
    parser = argparse.ArgumentParser(description="Torch MLP multimodal fusion residual for SMP video task 6b.")
    parser.add_argument("--data-dir", default="video-data")
    parser.add_argument("--oof", default="outputs/oof/power_tabular_v1_fix_oof.csv")
    parser.add_argument("--oof-pred-col", default="pred_blend")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--embedding-csvs", nargs="+", required=True)
    parser.add_argument("--feature-csvs", nargs="*", default=["outputs/features/video_asr.csv", "outputs/features/frame_ocr_rapid.csv", "outputs/features/video_stats.csv", "outputs/features/video_file_props.csv", "outputs/features/amcfg_anchor_dense_ocr_asr_features.csv"])
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--shrinks", default="0.01,0.02,0.04,0.06,0.08")
    parser.add_argument("--clip-log", type=float, default=0.8)
    parser.add_argument("--orth-anchor")
    parser.add_argument("--orth-source")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--name-prefix", default="candidate_278_torch_fusion")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/torch_fusion_report.csv")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    train, test, emb_meta = prepare_frames(args)
    cols = numeric_cols(train)
    shrinks = parse_float_list(args.shrinks)
    seeds = [int(float(x)) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for split_name, (hist_months, val_months) in SPLITS.items():
        hist = train[train["month_key"].isin(hist_months)].copy()
        val = train[train["month_key"].isin(val_months)].copy()
        x_hist, scaler = make_matrix(hist, hist, cols)
        x_val, _ = make_matrix(hist, val, cols, scaler)
        y_hist = np.log(np.clip(hist["popularity"].to_numpy(float), 1e-6, None) / np.clip(hist["oof_pred"].to_numpy(float), 1e-6, None))
        pred_logs = []
        for seed in seeds:
            pred_logs.append(train_predict(x_hist, y_hist, x_val, args, seed))
        pred_log = np.clip(np.mean(pred_logs, axis=0), -args.clip_log, args.clip_log)
        y = val["popularity"].to_numpy(float)
        anchor = val["oof_pred"].to_numpy(float)
        base_m = mape(y, anchor)
        for shrink in shrinks:
            pred = anchor * np.exp(shrink * pred_log)
            rows.append({"split": split_name, "shrink": shrink, "gain": base_m - mape(y, pred), "mape": mape(y, pred)})

    split_report = pd.DataFrame(rows)
    report = (
        split_report.groupby("shrink")
        .agg(mean_gain=("gain", "mean"), min_gain=("gain", "min"), mean_mape=("mape", "mean"))
        .reset_index()
        .sort_values(["mean_gain", "min_gain"], ascending=False)
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    split_report.to_csv(report_path.with_name(report_path.stem + "_splits.csv"), index=False)

    base = read_submission(args.base_submission)
    test = test.set_index("pid").loc[base.index].reset_index()
    x_train, scaler = make_matrix(train, train, cols)
    x_test, _ = make_matrix(train, test, cols, scaler)
    y_train = np.log(np.clip(train["popularity"].to_numpy(float), 1e-6, None) / np.clip(train["oof_pred"].to_numpy(float), 1e-6, None))
    pred_logs = []
    for seed in seeds:
        pred_logs.append(train_predict(x_train, y_train, x_test, args, seed + 100))
    pred_log = np.clip(np.mean(pred_logs, axis=0), -args.clip_log, args.clip_log)
    if args.orth_anchor and args.orth_source:
        orth_anchor, orth_source = aligned_values([read_submission(args.orth_anchor), read_submission(args.orth_source)])
        orth_anchor = orth_anchor.loc[base.index]
        orth_source = orth_source.loc[base.index]
        basis = np.log(np.clip(orth_source.to_numpy(float), 1e-6, None) / np.clip(orth_anchor.to_numpy(float), 1e-6, None))
        pred_log = project_out(pred_log, basis)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for shrink in shrinks:
        pred = base.to_numpy(float) * np.exp(shrink * pred_log)
        path = out_dir / f"{args.name_prefix}_s{str(shrink).replace('.', 'p')}.csv"
        write_submission(base.index, pred, path)
        written.append(str(path))

    meta = {
        "base_submission": args.base_submission,
        "embedding_meta": emb_meta,
        "features": len(cols),
        "written": written,
        "pred_log_stats": {
            "mean": float(np.mean(pred_log)),
            "std": float(np.std(pred_log)),
            "mad": float(np.mean(np.abs(pred_log))),
            "min": float(np.min(pred_log)),
            "max": float(np.max(pred_log)),
        },
    }
    report_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(report.to_string(index=False))
    print("Pred log stats:", meta["pred_log_stats"])
    print("Written candidates:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
