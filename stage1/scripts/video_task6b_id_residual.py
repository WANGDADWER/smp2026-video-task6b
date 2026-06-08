from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smp26.data import load_train_test
from smp26.features import numeric_feature_columns, prepare_features
from smp26.metrics import mape
from video_task6b_residual_lab import read_submission, write_submission


SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}


def id_number(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def add_id_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pid_num"] = id_number(out["pid"])
    out["vid_num"] = id_number(out["vid"])
    out["uid_num"] = id_number(out["uid"])
    out["pid_vid_gap"] = out["vid_num"] - out["pid_num"]
    out["pid_uid_gap"] = out["pid_num"] - out["uid_num"]
    out["vid_uid_gap"] = out["vid_num"] - out["uid_num"]
    out["pid_per_uid"] = out["pid_num"] / (out["uid_num"] + 1)
    out["vid_per_uid"] = out["vid_num"] / (out["uid_num"] + 1)
    out["month_index"] = out["dt"].dt.year * 12 + out["dt"].dt.month
    out["day_index"] = (out["dt"] - out["dt"].min()).dt.total_seconds().fillna(0) / 86400.0
    out["pid_month_interaction"] = out["pid_num"] * out["month_index"]
    out["uid_month_interaction"] = out["uid_num"] * out["month_index"]
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    id_cols = [
        "pid_num",
        "vid_num",
        "uid_num",
        "pid_vid_gap",
        "pid_uid_gap",
        "vid_uid_gap",
        "pid_per_uid",
        "vid_per_uid",
        "month_index",
        "day_index",
        "pid_month_interaction",
        "uid_month_interaction",
    ]
    base_cols = [
        c
        for c in numeric_feature_columns(df)
        if c
        in {
            "user_following_count",
            "user_follower_count",
            "user_likes_count",
            "user_video_count",
            "user_digg_count",
            "user_heart_count",
            "user_friend_count",
            "video_height",
            "video_width",
            "video_duration",
            "music_duration",
            "log1p_user_follower_count",
            "log1p_user_likes_count",
            "log1p_user_likes_count_fixed",
            "log1p_user_video_count",
            "log1p_user_digg_count",
            "log1p_user_heart_count",
            "log1p_user_heart_count_fixed",
            "log1p_user_friend_count",
            "likes_per_video",
            "heart_per_video",
            "followers_per_video",
            "digg_per_video",
            "heart_per_follower",
            "likes_per_follower",
            "following_follower_ratio",
            "follower_per_following",
            "likes_per_digg",
            "followers_per_digg",
            "log1p_likes_per_video",
            "log1p_heart_per_video",
            "log1p_followers_per_video",
            "log1p_digg_per_video",
            "log1p_heart_per_follower",
            "log1p_likes_per_follower",
            "log1p_following_follower_ratio",
            "log1p_follower_per_following",
            "log1p_likes_per_digg",
            "log1p_followers_per_digg",
            "content_text_len",
            "full_text_len",
            "suggested_count",
            "is_original_sound",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        }
    ]
    return [c for c in id_cols + base_cols if c in df.columns]


def make_model(kind: str, alpha: float, seed: int):
    if kind == "ridge_poly":
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=alpha),
        )
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=500,
            max_leaf_nodes=15,
            l2_regularization=alpha,
            random_state=seed,
            loss="absolute_error",
        )
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=max(5, int(alpha)),
            max_features=0.75,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(kind)


def fit_predict_residual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    kind: str,
    alpha: float,
    seed: int,
) -> np.ndarray:
    x_train = train_df[cols].fillna(train_df[cols].median(numeric_only=True)).to_numpy(float)
    x_test = test_df[cols].fillna(train_df[cols].median(numeric_only=True)).to_numpy(float)
    y = train_df["log_resid"].to_numpy(float)
    model = make_model(kind, alpha, seed)
    model.fit(x_train, y)
    return model.predict(x_test)


def parse_list(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ID/time-structure residual vectors for SMP video task 6b.")
    parser.add_argument("--data-dir", default="video-data")
    parser.add_argument("--oof", default="outputs/oof/power_tabular_v1_fix_oof.csv")
    parser.add_argument("--oof-pred-col", default="pred_blend")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--models", default="ridge_poly,ridge,hgb,extra")
    parser.add_argument("--alphas", default="1,10,100")
    parser.add_argument("--shrinks", default="0.05,0.1,0.2,0.35")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name-prefix", default="candidate_229_id_resid")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/id_residual_report.csv")
    args = parser.parse_args()

    train, test = load_train_test(args.data_dir)
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    train = add_id_features(prepare_features(train, train_min_time=train_min_time))
    test = add_id_features(prepare_features(test, train_min_time=train_min_time))
    oof = pd.read_csv(args.oof)
    train = train.merge(
        oof[["pid", args.oof_pred_col]].rename(columns={args.oof_pred_col: "oof_pred"}),
        on="pid",
        how="inner",
    )
    train["oof_pred"] = np.clip(train["oof_pred"].astype(float), 1e-6, None)
    train["log_resid"] = np.log(np.clip(train["popularity"].astype(float), 1e-6, None) / train["oof_pred"])
    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    cols = feature_columns(train)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    alphas = parse_list(args.alphas, float)
    shrinks = parse_list(args.shrinks, float)
    rows = []
    split_preds: dict[tuple[str, float], list[float]] = {}
    for kind in models:
        for alpha in alphas:
            gains = []
            base_mapes = []
            max_abs_resid = []
            for split_name, (hist_months, val_months) in SPLITS.items():
                hist = train[train["month_key"].isin(hist_months)].copy()
                val = train[train["month_key"].isin(val_months)].copy()
                if len(hist) == 0 or len(val) == 0:
                    continue
                pred_log_resid = fit_predict_residual(hist, val, cols, kind, alpha, args.seed)
                base_pred = val["oof_pred"].to_numpy(float)
                y = val["popularity"].to_numpy(float)
                base_m = mape(y, base_pred)
                best_gain = -999.0
                best_shrink = 0.0
                for shrink in shrinks:
                    pred = base_pred * np.exp(shrink * pred_log_resid)
                    gain = base_m - mape(y, pred)
                    if gain > best_gain:
                        best_gain = gain
                        best_shrink = shrink
                gains.append(best_gain)
                base_mapes.append(base_m)
                max_abs_resid.append(float(np.max(np.abs(pred_log_resid))))
            if gains:
                rows.append(
                    {
                        "kind": kind,
                        "alpha": alpha,
                        "mean_gain": float(np.mean(gains)),
                        "min_gain": float(np.min(gains)),
                        "mean_base_mape": float(np.mean(base_mapes)),
                        "max_abs_log_resid": float(np.max(max_abs_resid)),
                        "features": len(cols),
                    }
                )

    report = pd.DataFrame(rows).sort_values(["mean_gain", "min_gain"], ascending=False)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    best = report.head(3)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = read_submission(args.base_submission)
    test = test.set_index("pid").loc[base.index].reset_index()
    written = []
    for i, row in enumerate(best.itertuples(index=False), start=1):
        pred_log_resid = fit_predict_residual(train, test, cols, row.kind, float(row.alpha), args.seed)
        pred_log_resid = np.clip(pred_log_resid, -1.0, 1.0)
        for shrink in shrinks:
            pred = base.to_numpy(float) * np.exp(shrink * pred_log_resid)
            label_alpha = str(row.alpha).replace(".", "p")
            label_shrink = str(shrink).replace(".", "p")
            path = out_dir / f"{args.name_prefix}_{i}_{row.kind}_a{label_alpha}_s{label_shrink}.csv"
            write_submission(base.index, pred, path)
            written.append(str(path))

    meta = {
        "base_submission": args.base_submission,
        "features": cols,
        "report": str(report_path),
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
