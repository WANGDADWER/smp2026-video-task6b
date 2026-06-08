from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from video_task6b_residual_lab import aligned_values, read_submission, write_submission


def parse_basis_pair(text: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in text.split(",", 2)]
    if len(parts) != 3:
        raise ValueError("--basis-pair must be name,anchor_path,source_path")
    return parts[0], parts[1], parts[2]


def parse_range(text: str) -> tuple[str, np.ndarray]:
    name, lo, hi, steps = [p.strip() for p in text.split(":")]
    return name, np.linspace(float(lo), float(hi), int(steps))


def make_features(coords: np.ndarray, interactions: bool) -> tuple[np.ndarray, list[str]]:
    n, d = coords.shape
    cols = [np.ones(n)]
    names = ["intercept"]
    for j in range(d):
        cols.append(coords[:, j])
        names.append(f"x{j}")
    for j in range(d):
        cols.append(coords[:, j] ** 2)
        names.append(f"x{j}^2")
    if interactions:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(coords[:, i] * coords[:, j])
                names.append(f"x{i}*x{j}")
    return np.vstack(cols).T, names


def weighted_ridge_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    sqrt_w = np.sqrt(weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


def mape_proxy_distance(pred: np.ndarray, current: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - current) / np.clip(np.abs(current), 1e-6, None)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Use public LB feedback as a low-dimensional optimizer.")
    parser.add_argument("--scores-csv", default="outputs/video_task6b/reports/public_scores_video_task6b.csv")
    parser.add_argument("--anchor-submission", required=True)
    parser.add_argument("--current-submission", required=True)
    parser.add_argument("--basis-pair", action="append", required=True)
    parser.add_argument("--search-range", action="append", required=True)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--weight-temperature", type=float, default=0.0015)
    parser.add_argument("--trust-penalty", type=float, default=0.02)
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--write-k", type=int, default=4)
    parser.add_argument("--name-prefix", default="candidate_221_pfopt")
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/public_feedback_optimizer_report.csv")
    args = parser.parse_args()

    basis_defs = [parse_basis_pair(x) for x in args.basis_pair]
    basis_names = [x[0] for x in basis_defs]
    range_defs = dict(parse_range(x) for x in args.search_range)
    missing_ranges = [name for name in basis_names if name not in range_defs]
    if missing_ranges:
        raise ValueError(f"Missing search ranges for basis: {missing_ranges}")

    anchor = read_submission(args.anchor_submission)
    current = read_submission(args.current_submission)
    basis_series = []
    for _, anchor_path, source_path in basis_defs:
        basis_series.extend([read_submission(anchor_path), read_submission(source_path)])
    aligned = aligned_values([anchor, current] + basis_series)
    anchor = aligned[0]
    current = aligned[1]

    bases = []
    pos = 2
    for _name, _anchor_path, _source_path in basis_defs:
        b_anchor = aligned[pos]
        b_source = aligned[pos + 1]
        pos += 2
        bases.append((b_source - b_anchor).to_numpy(dtype=float))
    basis_matrix = np.vstack(bases).T

    scores = pd.read_csv(args.scores_csv)
    rows = []
    preds = []
    for row in scores.itertuples(index=False):
        path = Path(str(getattr(row, "path", "")))
        if not str(path) or not path.exists():
            continue
        try:
            sub = read_submission(path).loc[anchor.index]
        except Exception:
            continue
        delta = sub.to_numpy(dtype=float) - anchor.to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(basis_matrix, delta, rcond=None)
        recon = basis_matrix @ coef
        rows.append(
            {
                "candidate": str(getattr(row, "candidate")),
                "public_mape": float(getattr(row, "public_mape")),
                "path": str(path),
                "recon_mad": float(np.mean(np.abs(delta - recon))),
                **{f"coef_{name}": float(coef[i]) for i, name in enumerate(basis_names)},
            }
        )
        preds.append(sub.to_numpy(dtype=float))

    train = pd.DataFrame(rows)
    if len(train) < len(basis_names) + 4:
        raise ValueError("Not enough scored submissions with existing files.")

    coords = train[[f"coef_{name}" for name in basis_names]].to_numpy(dtype=float)
    y = train["public_mape"].to_numpy(dtype=float)
    best = float(np.min(y))
    weights = np.exp(-(y - best) / args.weight_temperature)
    weights = np.clip(weights, 1e-4, 1.0)
    x, feature_names = make_features(coords, args.interactions)
    beta = weighted_ridge_fit(x, y, weights, args.alpha)

    current_delta = current.to_numpy(dtype=float) - anchor.to_numpy(dtype=float)
    current_coef, *_ = np.linalg.lstsq(basis_matrix, current_delta, rcond=None)

    grids = [range_defs[name] for name in basis_names]
    search_rows = []
    for values in product(*grids):
        coord = np.asarray(values, dtype=float)
        xx, _ = make_features(coord.reshape(1, -1), args.interactions)
        pred_score = float((xx @ beta)[0])
        pred = anchor.to_numpy(dtype=float) + basis_matrix @ coord
        trust = mape_proxy_distance(pred, current.to_numpy(dtype=float))
        objective = pred_score + args.trust_penalty * trust
        search_rows.append(
            {
                "objective": objective,
                "surrogate_mape": pred_score,
                "trust_distance": trust,
                **{f"coef_{name}": float(coord[i]) for i, name in enumerate(basis_names)},
            }
        )

    search = pd.DataFrame(search_rows).sort_values(["objective", "surrogate_mape"])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    used = set()
    for i, row in search.head(args.top_k * 5).iterrows():
        coord = np.asarray([row[f"coef_{name}"] for name in basis_names], dtype=float)
        # Avoid writing near-duplicates in coefficient space.
        rounded = tuple(np.round(coord, 4))
        if rounded in used:
            continue
        used.add(rounded)
        if len(written) >= args.write_k:
            break
        pred = anchor.to_numpy(dtype=float) + basis_matrix @ coord
        label = "_".join(f"{name}{coord[j]:+.3f}".replace(".", "p").replace("+", "p").replace("-", "m") for j, name in enumerate(basis_names))
        path = out_dir / f"{args.name_prefix}_{len(written)+1}_{label}.csv"
        write_submission(anchor.index, pred, path)
        written.append(str(path))

    payload = {
        "basis_names": basis_names,
        "feature_names": feature_names,
        "beta": beta.tolist(),
        "current_coefficients": {name: float(current_coef[i]) for i, name in enumerate(basis_names)},
        "best_seen_public_mape": best,
        "written": written,
    }
    with report_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    train.sort_values("public_mape").to_csv(report_path, index=False)
    search.head(200).to_csv(report_path.with_name(report_path.stem + "_search_top.csv"), index=False)
    try:
        with pd.ExcelWriter(report_path.with_suffix(".xlsx")) as writer:
            train.sort_values("public_mape").to_excel(writer, sheet_name="scored_coords", index=False)
            search.head(200).to_excel(writer, sheet_name="search_top", index=False)
    except Exception:
        pass

    print(f"Wrote report to {report_path}")
    print("Current coefficients:", payload["current_coefficients"])
    print(search.head(args.top_k).to_string(index=False))
    print("Written candidates:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
