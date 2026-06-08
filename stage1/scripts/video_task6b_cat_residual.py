from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape
from video_task6b_residual_lab import aligned_values, read_submission, write_submission


SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}


CAT_COLS = [
    "uid",
    "post_location",
    "post_text_language",
    "video_ratio",
    "video_format",
    "music_title",
    "month_str",
    "hour_str",
    "dow_str",
    "duration_bucket",
    "music_duration_bucket",
    "followers_bucket",
    "uid_freq_bucket",
    "pid_bin",
    "vid_bin",
    "uid_bin",
    "asr_language",
    "asr_present",
    "ocr_present",
]

TEXT_COLS = [
    "content_text",
    "suggested_text",
    "full_text",
    "asr_text",
    "ocr_text",
    "blip_caption",
]


def id_number(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def bucketize(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)


def qbucket(train_values: pd.Series, values: pd.Series, prefix: str, q: int = 8) -> pd.Series:
    clean = pd.to_numeric(train_values, errors="coerce").dropna()
    if clean.empty:
        return pd.Series(f"{prefix}_all", index=values.index)
    edges = np.unique(np.nanquantile(clean, np.linspace(0, 1, q + 1)))
    if len(edges) <= 2:
        return pd.Series(f"{prefix}_all", index=values.index)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"{prefix}_{i}" for i in range(len(edges) - 1)]
    return pd.cut(pd.to_numeric(values, errors="coerce"), edges, labels=labels, include_lowest=True).astype(str)


def attach_feature_csvs(
    train: pd.DataFrame, test: pd.DataFrame, paths: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for path in paths:
        feat = pd.read_csv(path)
        if "pid" not in feat.columns:
            raise ValueError(f"{path} must contain pid")
        drop = {"uid", "vid", "split", "video_path", "video_abspath"}
        keep = [col for col in feat.columns if col == "pid" or col not in drop]
        feat = feat[keep].copy()
        rename = {}
        prefix = Path(path).stem
        for col in feat.columns:
            if col == "pid":
                continue
            if col in train_out.columns and col not in {"asr_language", "asr_text", "ocr_text", "blip_caption"}:
                rename[col] = f"{prefix}_{col}"
        if rename:
            feat = feat.rename(columns=rename)
        train_out = train_out.merge(feat, on="pid", how="left")
        test_out = test_out.merge(feat, on="pid", how="left")
    return train_out, test_out


def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    combined_uid_count = pd.concat([train["uid"], test["uid"]]).astype(str).value_counts()
    for frame in [train, test]:
        frame["pid_num"] = id_number(frame["pid"])
        frame["vid_num"] = id_number(frame["vid"])
        frame["uid_num"] = id_number(frame["uid"])
        frame["pid_vid_gap"] = frame["vid_num"] - frame["pid_num"]
        frame["pid_uid_gap"] = frame["pid_num"] - frame["uid_num"]
        frame["vid_uid_gap"] = frame["vid_num"] - frame["uid_num"]
        frame["pid_per_uid"] = frame["pid_num"] / (frame["uid_num"] + 1)
        frame["vid_per_uid"] = frame["vid_num"] / (frame["uid_num"] + 1)
        frame["month_index"] = frame["dt"].dt.year * 12 + frame["dt"].dt.month
        frame["day_index"] = (frame["dt"] - train["dt"].min()).dt.total_seconds().fillna(0) / 86400.0
        frame["month_str"] = frame["dt"].dt.month.fillna(0).astype(int).astype(str)
        frame["hour_str"] = frame["dt"].dt.hour.fillna(0).astype(int).astype(str)
        frame["dow_str"] = frame["dt"].dt.dayofweek.fillna(0).astype(int).astype(str)
        frame["duration_bucket"] = bucketize(
            frame["video_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["dur_10", "dur_30", "dur_60", "dur_120", "dur_120p"],
        )
        frame["music_duration_bucket"] = bucketize(
            frame["music_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["music_10", "music_30", "music_60", "music_120", "music_120p"],
        )
        frame["uid_freq_bucket"] = bucketize(
            frame["uid"].astype(str).map(combined_uid_count).fillna(1),
            [-np.inf, 1, 2, 4, np.inf],
            ["uid_1", "uid_2", "uid_4", "uid_5p"],
        )
        if "asr_language" in frame.columns:
            frame["asr_language"] = frame["asr_language"].fillna("").astype(str)
            frame.loc[frame["asr_language"].isin(["", "nan", "None"]), "asr_language"] = "no_asr"
        else:
            frame["asr_language"] = "no_asr"
        if "asr_char_count" in frame.columns:
            frame["asr_present"] = (pd.to_numeric(frame["asr_char_count"], errors="coerce").fillna(0) > 0).map({True: "yes", False: "no"})
        else:
            frame["asr_present"] = "no"
        if "ocr_raw_text_count" in frame.columns:
            frame["ocr_present"] = (pd.to_numeric(frame["ocr_raw_text_count"], errors="coerce").fillna(0) > 0).map({True: "yes", False: "no"})
        else:
            frame["ocr_present"] = "no"
    for source, name in [
        ("user_follower_count", "followers"),
        ("pid_num", "pid"),
        ("vid_num", "vid"),
        ("uid_num", "uid"),
    ]:
        train[f"{name}_bin"] = qbucket(train[source], train[source], name)
        test[f"{name}_bin"] = qbucket(train[source], test[source], name)
    for frame in [train, test]:
        frame["followers_bucket"] = frame.get("followers_bin", "followers_all").astype(str)
    for df in [train, test]:
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return train, test


def model_columns(df: pd.DataFrame, use_text_features: bool) -> tuple[list[str], list[str], list[str]]:
    skip = {
        "pid",
        "vid",
        "post_time",
        "video_path",
        "post_content",
        "post_suggested_words",
        "dt",
        "split",
        "popularity",
        "oof_pred",
        "log_resid",
        "month_key",
    }
    dynamic_cats = [c for c in df.columns if c.endswith("_cluster")]
    cats = [c for c in CAT_COLS + dynamic_cats if c in df.columns]
    texts = [c for c in TEXT_COLS if use_text_features and c in df.columns and c not in cats]
    nums = []
    for col in df.columns:
        if col in skip or col in cats or col in texts:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            nums.append(col)
    return nums + cats + texts, cats, texts


def prepare_pool(
    train_df: pd.DataFrame,
    cols: list[str],
    cats: list[str],
    texts: list[str],
    reference: pd.DataFrame | None = None,
    label: np.ndarray | None = None,
) -> Pool:
    out = train_df[cols].copy()
    ref = train_df if reference is None else reference
    for col in cols:
        if col in cats:
            out[col] = out[col].fillna("__NA__").astype(str)
        elif col in texts:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(pd.to_numeric(ref[col], errors="coerce").median())
    cat_idx = [cols.index(c) for c in cats]
    text_idx = [cols.index(c) for c in texts]
    return Pool(out, label=label, cat_features=cat_idx, text_features=text_idx)


def fit_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cols: list[str],
    cats: list[str],
    texts: list[str],
    depth: int,
    l2: float,
    seed: int,
    loss: str,
    iterations: int,
) -> np.ndarray:
    train_pool = prepare_pool(train_df, cols, cats, texts, label=train_df["log_resid"].to_numpy(float))
    target_pool = prepare_pool(target_df, cols, cats, texts, reference=train_df)
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        learning_rate=0.035,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    return model.predict(target_pool)


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    b = basis - float(np.mean(basis))
    vv = v - float(np.mean(v))
    denom = float(np.dot(b, b))
    if denom == 0:
        return v
    beta = float(np.dot(vv, b) / denom)
    return v - beta * basis


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost categorical residual model for SMP video task 6b.")
    parser.add_argument("--data-dir", default="video-data")
    parser.add_argument("--oof", default="outputs/oof/power_tabular_v1_fix_oof.csv")
    parser.add_argument("--oof-pred-col", default="pred_blend")
    parser.add_argument("--feature-csvs", nargs="*", default=["outputs/features/video_asr.csv", "outputs/features/frame_ocr_rapid.csv"])
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--orth-anchor")
    parser.add_argument("--orth-source")
    parser.add_argument("--depths", default="4,6")
    parser.add_argument("--l2s", default="10,30,100")
    parser.add_argument("--losses", default="MAE,RMSE")
    parser.add_argument("--shrinks", default="0.01,0.02,0.035,0.05")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--use-text-features", action="store_true")
    parser.add_argument("--name-prefix", default="candidate_232_cat_resid")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/cat_residual_report.csv")
    args = parser.parse_args()

    train, test = load_train_test(args.data_dir)
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)
    train, test = attach_feature_csvs(train, test, args.feature_csvs)
    train, test = add_features(train, test)
    oof = pd.read_csv(args.oof)
    train = train.merge(
        oof[["pid", args.oof_pred_col]].rename(columns={args.oof_pred_col: "oof_pred"}),
        on="pid",
        how="inner",
    )
    train["oof_pred"] = np.clip(train["oof_pred"].astype(float), 1e-6, None)
    train["log_resid"] = np.log(np.clip(train["popularity"].astype(float), 1e-6, None) / train["oof_pred"])
    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    cols, cats, texts = model_columns(train, args.use_text_features)

    shrinks = parse_float_list(args.shrinks)
    rows = []
    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        for l2 in parse_float_list(args.l2s):
            for loss in [x.strip() for x in args.losses.split(",") if x.strip()]:
                gains = []
                best_shrinks = []
                for split_name, (hist_months, val_months) in SPLITS.items():
                    hist = train[train["month_key"].isin(hist_months)].copy()
                    val = train[train["month_key"].isin(val_months)].copy()
                    if len(hist) == 0 or len(val) == 0:
                        continue
                    pred_log = np.clip(
                        fit_predict(hist, val, cols, cats, texts, depth, l2, args.seed, loss, args.iterations),
                        -1.0,
                        1.0,
                    )
                    base_pred = val["oof_pred"].to_numpy(float)
                    y = val["popularity"].to_numpy(float)
                    base_m = mape(y, base_pred)
                    split_best_gain = -999.0
                    split_best_shrink = 0.0
                    for shrink in shrinks:
                        pred = base_pred * np.exp(shrink * pred_log)
                        gain = base_m - mape(y, pred)
                        if gain > split_best_gain:
                            split_best_gain = gain
                            split_best_shrink = shrink
                    gains.append(split_best_gain)
                    best_shrinks.append(split_best_shrink)
                rows.append(
                    {
                        "depth": depth,
                        "l2": l2,
                        "loss": loss,
                        "mean_gain": float(np.mean(gains)),
                        "min_gain": float(np.min(gains)),
                        "best_shrink_median": float(np.median(best_shrinks)),
                        "features": len(cols),
                        "cats": len(cats),
                        "texts": len(texts),
                    }
                )

    report = pd.DataFrame(rows).sort_values(["mean_gain", "min_gain"], ascending=False)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    base = read_submission(args.base_submission)
    test = test.set_index("pid").loc[base.index].reset_index()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basis = None
    if args.orth_anchor and args.orth_source:
        orth_anchor, orth_source = aligned_values([read_submission(args.orth_anchor), read_submission(args.orth_source)])
        orth_anchor = orth_anchor.loc[base.index]
        orth_source = orth_source.loc[base.index]
        basis = np.log(np.clip(orth_source.to_numpy(float), 1e-6, None) / np.clip(orth_anchor.to_numpy(float), 1e-6, None))

    written = []
    for i, row in enumerate(report.head(3).itertuples(index=False), start=1):
        pred_log = np.clip(
            fit_predict(train, test, cols, cats, texts, int(row.depth), float(row.l2), args.seed, row.loss, args.iterations),
            -1.0,
            1.0,
        )
        if basis is not None:
            pred_log = project_out(pred_log, basis)
        for shrink in shrinks:
            pred = base.to_numpy(float) * np.exp(shrink * pred_log)
            label_l2 = str(row.l2).replace(".", "p")
            label_shrink = str(shrink).replace(".", "p").replace("-", "m")
            path = out_dir / f"{args.name_prefix}_{i}_d{row.depth}_l2{label_l2}_{row.loss}_s{label_shrink}.csv"
            write_submission(base.index, pred, path)
            written.append(str(path))

    meta = {
        "base_submission": args.base_submission,
        "orth_anchor": args.orth_anchor,
        "orth_source": args.orth_source,
        "features": cols,
        "cat_features": cats,
        "text_features": texts,
        "written": written,
    }
    report_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    print(report.head(12).to_string(index=False))
    print("Written candidates:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
