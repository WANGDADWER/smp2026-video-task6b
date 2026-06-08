from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape
from video_task6b_cat_residual import attach_feature_csvs
from video_task6b_prototype_residual import parse_float_list
from video_task6b_residual_lab import read_submission, write_submission


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}")
STOP = {
    "the",
    "and",
    "for",
    "you",
    "your",
    "with",
    "that",
    "this",
    "there",
    "are",
    "was",
    "has",
    "have",
    "from",
    "into",
    "some",
    "kind",
    "video",
    "fyp",
    "foryou",
    "foryoupage",
    "viral",
    "tiktok",
    "capcut",
    "follow",
    "like",
    "part",
}


def tokenize_text(text: str, max_tokens: int) -> list[str]:
    text = str(text).lower().replace("|", " ").replace(",", " ")
    toks = [t for t in TOKEN_RE.findall(text) if t not in STOP and not t.isdigit()]
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return toks


def row_tokens(row: pd.Series, max_tokens_per_field: int) -> Counter:
    fields = [
        ("content_text", 2.0),
        ("suggested_text", 1.5),
        ("music_text", 1.0),
        ("asr_text", 1.2),
        ("ocr_text", 1.3),
        ("blip_caption", 0.8),
    ]
    out: Counter = Counter()
    for col, weight in fields:
        if col not in row or pd.isna(row[col]):
            continue
        for tok in tokenize_text(str(row[col]), max_tokens_per_field):
            out[tok] += weight
    return out


def add_text_features(train: pd.DataFrame, test: pd.DataFrame, feature_csvs: list[str]):
    train, test = attach_feature_csvs(train, test, feature_csvs)
    for frame in [train, test]:
        for col in ["asr_text", "ocr_text", "blip_caption"]:
            if col in frame.columns:
                frame[col] = frame[col].fillna("").astype(str)
            else:
                frame[col] = ""
    return train, test


def token_stats(df: pd.DataFrame, tokens: list[Counter], smoothing: float, min_count: int, recent_weight_days: float = 0.0):
    y = np.log(np.clip(df["popularity"].to_numpy(float), 1e-6, None) / np.clip(df["anchor"].to_numpy(float), 1e-6, None))
    if recent_weight_days > 0:
        max_dt = df["dt"].max()
        ages = (max_dt - df["dt"]).dt.total_seconds().fillna(0).to_numpy() / 86400.0
        row_w = np.exp(-ages / recent_weight_days)
    else:
        row_w = np.ones(len(df), dtype=float)
    cnt = defaultdict(float)
    sum_res = defaultdict(float)
    doc_cnt = defaultdict(int)
    for i, ctr in enumerate(tokens):
        if not ctr:
            continue
        seen = set()
        for tok, val in ctr.items():
            w = float(val) * row_w[i]
            cnt[tok] += w
            sum_res[tok] += w * y[i]
            if tok not in seen:
                doc_cnt[tok] += 1
                seen.add(tok)
    n_docs = max(len(df), 1)
    global_mean = float(np.average(y, weights=row_w))
    stats = {}
    for tok, c in cnt.items():
        if doc_cnt[tok] < min_count:
            continue
        mean = sum_res[tok] / max(c, 1e-9)
        shrink = doc_cnt[tok] / (doc_cnt[tok] + smoothing)
        idf = math.log((n_docs + 2.0) / (doc_cnt[tok] + 1.0))
        stats[tok] = {
            "score": shrink * mean + (1.0 - shrink) * global_mean,
            "idf": max(idf, 0.0),
            "count": doc_cnt[tok],
        }
    return stats, global_mean


def score_rows(tokens: list[Counter], stats: dict, default: float, idf_power: float, count_power: float):
    out = np.zeros(len(tokens), dtype=float)
    coverage = np.zeros(len(tokens), dtype=float)
    for i, ctr in enumerate(tokens):
        num = 0.0
        den = 0.0
        for tok, val in ctr.items():
            item = stats.get(tok)
            if item is None:
                continue
            w = float(val) * ((item["idf"] + 1e-3) ** idf_power) * ((item["count"] + 1.0) ** count_power)
            num += w * item["score"]
            den += w
        if den > 0:
            out[i] = num / den
            coverage[i] = den
        else:
            out[i] = default
    return out, coverage


def month_lt(series: pd.Series, month: str) -> pd.Series:
    return series < month


def main():
    parser = argparse.ArgumentParser(description="Token/hashtag trend residuals for SMP video task 6b.")
    parser.add_argument("--data-dir", default="video-data")
    parser.add_argument("--oof", default="outputs/oof/power_tabular_v1_fix_oof.csv")
    parser.add_argument("--oof-pred-col", default="pred_blend")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--feature-csvs", nargs="*", default=["outputs/features/video_asr.csv", "outputs/features/frame_ocr_rapid.csv", "outputs/features/blip_video_captions_f01234567_t24.csv"])
    parser.add_argument("--smoothings", default="5,10,20,50")
    parser.add_argument("--min-counts", default="2,3,5")
    parser.add_argument("--idf-powers", default="0,0.5,1")
    parser.add_argument("--count-powers", default="0,-0.25")
    parser.add_argument("--recent-days", default="0,90,180")
    parser.add_argument("--shrinks", default="0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--max-tokens-per-field", type=int, default=80)
    parser.add_argument("--clip-log", type=float, default=0.8)
    parser.add_argument("--name-prefix", default="candidate_273_token_trend")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/token_trend_report.csv")
    args = parser.parse_args()

    train, test = load_train_test(args.data_dir)
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)
    train, test = add_text_features(train, test, args.feature_csvs)
    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    test["month_key"] = test["dt"].dt.strftime("%Y-%m")
    oof = pd.read_csv(args.oof)
    train = train.merge(oof[["pid", args.oof_pred_col]].rename(columns={args.oof_pred_col: "oof_pred"}), on="pid", how="inner")
    train["anchor"] = np.clip(pd.to_numeric(train["oof_pred"], errors="coerce"), 1e-6, None)
    base = read_submission(args.base_submission)
    test = test.merge(base.rename("base_pred"), left_on="pid", right_index=True, how="inner")
    test["anchor"] = np.clip(test["base_pred"].to_numpy(float), 1e-6, None)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_tokens = [row_tokens(row, args.max_tokens_per_field) for _, row in train.iterrows()]
    test_tokens = [row_tokens(row, args.max_tokens_per_field) for _, row in test.iterrows()]

    smoothings = parse_float_list(args.smoothings)
    min_counts = [int(float(x)) for x in args.min_counts.split(",") if x.strip()]
    idf_powers = parse_float_list(args.idf_powers)
    count_powers = parse_float_list(args.count_powers)
    recent_days_list = parse_float_list(args.recent_days)
    shrinks = parse_float_list(args.shrinks)

    val_months = ["2023-05", "2023-06", "2023-07", "2023-08"]
    rows = []
    signals_cache = {}
    for smoothing in smoothings:
        for min_count in min_counts:
            for idf_power in idf_powers:
                for count_power in count_powers:
                    for recent_days in recent_days_list:
                        gains = {s: [] for s in shrinks}
                        scores = {s: [] for s in shrinks}
                        coverages = []
                        for month in val_months:
                            hist_mask = month_lt(train["month_key"], month).to_numpy()
                            val_mask = train["month_key"].eq(month).to_numpy()
                            hist = train.loc[hist_mask].copy()
                            val = train.loc[val_mask].copy()
                            hist_tokens = [tok for tok, keep in zip(train_tokens, hist_mask) if keep]
                            val_tokens = [tok for tok, keep in zip(train_tokens, val_mask) if keep]
                            stats, default = token_stats(hist, hist_tokens, smoothing, min_count, recent_days)
                            signal, cov = score_rows(val_tokens, stats, default, idf_power, count_power)
                            signal = np.clip(signal, -args.clip_log, args.clip_log)
                            coverages.append(float(np.mean(cov > 0)))
                            target = val["popularity"].to_numpy(float)
                            anchor = val["anchor"].to_numpy(float)
                            base_m = mape(target, anchor)
                            for shrink in shrinks:
                                pred = anchor * np.exp(shrink * signal)
                                score = mape(target, pred)
                                gains[shrink].append(base_m - score)
                                scores[shrink].append(score)
                        for shrink in shrinks:
                            rows.append(
                                {
                                    "smoothing": smoothing,
                                    "min_count": min_count,
                                    "idf_power": idf_power,
                                    "count_power": count_power,
                                    "recent_days": recent_days,
                                    "shrink": shrink,
                                    "mean_gain": float(np.mean(gains[shrink])),
                                    "min_gain": float(np.min(gains[shrink])),
                                    "mean_mape": float(np.mean(scores[shrink])),
                                    "coverage": float(np.mean(coverages)),
                                }
                            )

    report = pd.DataFrame(rows).sort_values(["mean_gain", "min_gain"], ascending=False)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = base.loc[test["pid"]]
    written = []
    for i, row in enumerate(report.head(10).to_dict("records"), start=1):
        stats, default = token_stats(
            train,
            train_tokens,
            float(row["smoothing"]),
            int(row["min_count"]),
            float(row["recent_days"]),
        )
        signal, cov = score_rows(test_tokens, stats, default, float(row["idf_power"]), float(row["count_power"]))
        signal = np.clip(signal, -args.clip_log, args.clip_log)
        pred = base.to_numpy(float) * np.exp(float(row["shrink"]) * signal)
        label = (
            f"{args.name_prefix}_{i}_sm{str(row['smoothing']).replace('.', 'p')}"
            f"_mc{int(row['min_count'])}_idf{str(row['idf_power']).replace('.', 'p')}"
            f"_cp{str(row['count_power']).replace('.', 'm').replace('-', 'm')}"
            f"_rd{str(row['recent_days']).replace('.', 'p')}_s{str(row['shrink']).replace('.', 'p')}.csv"
        )
        path = out_dir / label
        write_submission(base.index, pred, path)
        written.append(str(path))

    meta = {"base_submission": args.base_submission, "written": written}
    report_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(report.head(20).to_string(index=False))
    print("Written candidates:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
