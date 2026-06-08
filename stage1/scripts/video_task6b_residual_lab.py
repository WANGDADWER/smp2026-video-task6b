from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCORE_COL = "polularity_score"


def read_submission(path: str | Path) -> pd.Series:
    path = Path(path)
    df = pd.read_csv(path)
    if "pid" not in df.columns or SCORE_COL not in df.columns:
        raise ValueError(f"{path} must contain pid,{SCORE_COL}")
    if df["pid"].duplicated().any():
        dupes = df.loc[df["pid"].duplicated(), "pid"].head().tolist()
        raise ValueError(f"{path} has duplicated pid values: {dupes}")
    return df.set_index("pid")[SCORE_COL].astype(float)


def aligned_values(series: Iterable[pd.Series]) -> list[pd.Series]:
    series = list(series)
    if not series:
        return []
    index = series[0].index
    out = []
    for s in series:
        missing = index.difference(s.index)
        extra = s.index.difference(index)
        if len(missing) or len(extra):
            raise ValueError(
                f"pid mismatch: missing={len(missing)} extra={len(extra)}"
            )
        out.append(s.loc[index])
    return out


def weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    if w is None:
        return float(np.mean(x))
    return float(np.sum(x * w) / np.sum(w))


def centered(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    return x - weighted_mean(x, w)


def weighted_cosine(a: np.ndarray, b: np.ndarray, w: np.ndarray | None) -> float:
    aa = centered(a, w)
    bb = centered(b, w)
    if w is None:
        num = float(np.dot(aa, bb))
        den = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    else:
        num = float(np.dot(w * aa, bb))
        den = float(np.sqrt(np.dot(w * aa, aa) * np.dot(w * bb, bb)))
    return 0.0 if den == 0 else num / den


def projection_coeff(v: np.ndarray, basis: np.ndarray, w: np.ndarray | None) -> float:
    bb = centered(basis, w)
    vv = centered(v, w)
    if w is None:
        den = float(np.dot(bb, bb))
        num = float(np.dot(vv, bb))
    else:
        den = float(np.dot(w * bb, bb))
        num = float(np.dot(w * vv, bb))
    return 0.0 if den == 0 else num / den


def orthogonalize(
    v: np.ndarray, bases: list[np.ndarray], w: np.ndarray | None
) -> tuple[np.ndarray, list[float]]:
    out = v.astype(float).copy()
    coeffs: list[float] = []
    for b in bases:
        beta = projection_coeff(out, b, w)
        out = out - beta * b
        coeffs.append(beta)
    return out, coeffs


def vector_stats(v: np.ndarray) -> dict[str, float]:
    q = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "mad": float(np.mean(np.abs(v))),
        "min": float(np.min(v)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "max": float(np.max(v)),
    }


def safe_predictions(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 1e-6, None)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_submission(index: pd.Index, pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"pid": index, SCORE_COL: safe_predictions(pred)})
    out.to_csv(path, index=False)


def cmd_summarize(args: argparse.Namespace) -> None:
    base = read_submission(args.base_submission)
    best = read_submission(args.best_submission) if args.best_submission else None
    candidates = [(Path(p).stem, read_submission(p)) for p in args.candidate_submissions]

    all_series = [base]
    if best is not None:
        all_series.append(best)
    all_series.extend(s for _, s in candidates)
    all_series = aligned_values(all_series)

    base = all_series[0]
    pos = 1
    best = all_series[pos] if best is not None else None
    pos += 1 if best is not None else 0
    candidates = [(name, all_series[pos + i]) for i, (name, _) in enumerate(candidates)]

    weights = None
    if args.weighting == "inverse_base":
        weights = 1.0 / np.clip(base.to_numpy(), 1e-6, None)
        weights = weights / np.mean(weights)

    basis = None
    if best is not None:
        basis = best.to_numpy() - base.to_numpy()

    rows = []
    for name, s in candidates:
        v = s.to_numpy() - base.to_numpy()
        row = {"candidate": name, **vector_stats(v)}
        row["corr_pred_base"] = weighted_cosine(s.to_numpy(), base.to_numpy(), weights)
        if basis is not None:
            row["corr_with_best_vector"] = weighted_cosine(v, basis, weights)
            row["beta_to_best_vector"] = projection_coeff(v, basis, weights)
            resid, _ = orthogonalize(v, [basis], weights)
            row["orth_mad_after_best"] = float(np.mean(np.abs(resid)))
            row["orth_std_after_best"] = float(np.std(resid))
        rows.append(row)

    report = pd.DataFrame(rows).sort_values(
        ["orth_mad_after_best" if basis is not None else "mad", "mad"],
        ascending=False,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"Wrote summary to {output}")
    if len(report):
        print(report.head(args.top).to_string(index=False))


def cmd_generate(args: argparse.Namespace) -> None:
    base = read_submission(args.base_submission)
    anchor = read_submission(args.vector_anchor)
    source = read_submission(args.vector_source)
    series = [base, anchor, source]

    basis_pairs: list[tuple[str, str, float]] = []
    for text in args.basis_pair or []:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError("--basis-pair must be anchor_path,source_path[,sign]")
        sign = float(parts[2]) if len(parts) == 3 else 1.0
        basis_pairs.append((parts[0], parts[1], sign))
        series.extend([read_submission(parts[0]), read_submission(parts[1])])

    series = aligned_values(series)
    base, anchor, source = series[:3]
    pos = 3

    weights = None
    if args.weighting == "inverse_base":
        weights = 1.0 / np.clip(base.to_numpy(), 1e-6, None)
        weights = weights / np.mean(weights)

    v = args.sign * (source.to_numpy() - anchor.to_numpy())
    bases = []
    for _anchor_path, _source_path, sign in basis_pairs:
        b_anchor = series[pos]
        b_source = series[pos + 1]
        pos += 2
        bases.append(sign * (b_source.to_numpy() - b_anchor.to_numpy()))

    coeffs: list[float] = []
    raw_stats = vector_stats(v)
    if bases:
        v, coeffs = orthogonalize(v, bases, weights)
    vec_stats = vector_stats(v)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name_prefix
    scales = parse_float_list(args.scales)

    written = []
    for scale in scales:
        label = str(scale).replace("-", "m").replace(".", "p")
        path = out_dir / f"{prefix}_s{label}.csv"
        pred = base.to_numpy() + scale * v
        write_submission(base.index, pred, path)
        written.append(str(path))

    vector_path = Path(args.vector_output) if args.vector_output else None
    if vector_path is not None:
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"pid": base.index, "vector": v}).to_csv(vector_path, index=False)

    meta = {
        "base_submission": args.base_submission,
        "vector_anchor": args.vector_anchor,
        "vector_source": args.vector_source,
        "sign": args.sign,
        "basis_pair": args.basis_pair or [],
        "basis_coefficients": coeffs,
        "weighting": args.weighting,
        "raw_vector_stats": raw_stats,
        "final_vector_stats": vec_stats,
        "scales": scales,
        "written": written,
        "vector_output": str(vector_path) if vector_path else None,
    }
    meta_path = out_dir / f"{prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(written)} submissions to {out_dir}")
    print(json.dumps(meta, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Residual-vector lab for SMP 2026 video task 6b submissions."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summarize")
    p.add_argument("--base-submission", required=True)
    p.add_argument("--best-submission")
    p.add_argument("--candidate-submissions", nargs="+", required=True)
    p.add_argument("--weighting", choices=["none", "inverse_base"], default="none")
    p.add_argument("--output", required=True)
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("generate")
    p.add_argument("--base-submission", required=True)
    p.add_argument("--vector-anchor", required=True)
    p.add_argument("--vector-source", required=True)
    p.add_argument("--sign", type=float, default=1.0)
    p.add_argument(
        "--basis-pair",
        action="append",
        help="anchor_path,source_path[,sign]. The resulting vector is removed.",
    )
    p.add_argument("--weighting", choices=["none", "inverse_base"], default="none")
    p.add_argument("--scales", required=True)
    p.add_argument("--name-prefix", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vector-output")
    p.set_defaults(func=cmd_generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
