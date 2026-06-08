"""
Integrate retrieval residual into the best model's component framework.

Approach:
  1. Compute retrieval direction = log(retrieval_raw) - log(base_anchor)
  2. Optionally orthogonalize against existing best-model direction
  3. Blend at multiple scales: final = base * exp(scale * direction)

Usage:
  python scripts/integrate_retrieval.py \
    --base submissions/candidate_336_...csv \
    --retrieval retrieval_output/candidate_retrieval_resid_raw_prediction.csv \
    --scales "0.25,0.5,0.75,1.0,1.5,2.0" \
    --output-dir integrated_output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from video_task6b_residual_lab import (
    aligned_values,
    orthogonalize,
    projection_coeff,
    read_submission,
    vector_stats,
    weighted_cosine,
    write_submission,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Integrate retrieval residual into component framework."
    )
    parser.add_argument("--base", required=True, help="Current best submission (anchor)")
    parser.add_argument("--retrieval", required=True, help="Retrieval raw prediction CSV")
    parser.add_argument("--best-direction-anchor", help="If given, compute best direction = best_source - best_anchor")
    parser.add_argument("--best-direction-source", help="Pair with --best-direction-anchor")
    parser.add_argument("--scales", default="0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument("--name-prefix", default="candidate_retrieval")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and align
    base = read_submission(args.base)
    retrieval = read_submission(args.retrieval)

    all_series = [base, retrieval]
    best_anchor = best_source = None
    if args.best_direction_anchor and args.best_direction_source:
        best_anchor = read_submission(args.best_direction_anchor)
        best_source = read_submission(args.best_direction_source)
        all_series.extend([best_anchor, best_source])

    aligned = aligned_values(all_series)
    base = aligned[0]
    retrieval = aligned[1]

    # Compute retrieval direction in log space
    base_log = np.log(np.clip(base.to_numpy(float), 1e-6, None))
    retrieval_log = np.log(np.clip(retrieval.to_numpy(float), 1e-6, None))
    retrieval_direction = retrieval_log - base_log

    raw_stats = vector_stats(retrieval_direction)
    print("Raw retrieval direction stats:")
    for k, v in raw_stats.items():
        print(f"  {k}: {v:.6f}")

    # If best direction provided, compute relationships
    orthogonalized = False
    if best_anchor is not None and best_source is not None:
        best_anchor = aligned[2]
        best_source = aligned[3]
        best_anchor_log = np.log(np.clip(best_anchor.to_numpy(float), 1e-6, None))
        best_source_log = np.log(np.clip(best_source.to_numpy(float), 1e-6, None))
        best_direction = best_source_log - best_anchor_log

        # Correlation analysis
        cos_sim = weighted_cosine(retrieval_direction, best_direction, None)
        proj_coeff = projection_coeff(retrieval_direction, best_direction, None)
        print(f"\nRetrieval vs best direction:")
        print(f"  cosine similarity: {cos_sim:.6f}")
        print(f"  projection coeff:  {proj_coeff:.6f}")

        # For orthogonalization, we need list of bases
        bases = [best_direction]

        # Orthogonalize: remove best-direction component from retrieval
        retrieval_orth, orth_coeffs = orthogonalize(
            retrieval_direction, bases, None
        )
        orth_stats = vector_stats(retrieval_orth)
        print(f"\nAfter orthogonalization:")
        print(f"  removed coeff: {orth_coeffs[0]:.6f}")
        print(f"  remaining mad:  {orth_stats['mad']:.6f}")
        print(f"  remaining std:  {orth_stats['std']:.6f}")

        # Write both raw and orthogonalized versions
        scales = parse_float_list(args.scales)
        for use_orth, suffix in [(False, ""), (True, "_orth")]:
            direction = retrieval_orth if use_orth else retrieval_direction
            direction_centered = direction - float(np.mean(direction))

            for scale in scales:
                pred = np.exp(base_log + scale * direction_centered)
                label = str(scale).replace(".", "p").replace("-", "m")
                path = out_dir / f"{args.name_prefix}_s{label}{suffix}.csv"
                write_submission(base.index, pred, path)

        written_count = len(scales) * 2
    else:
        scales = parse_float_list(args.scales)
        direction_centered = retrieval_direction - float(np.mean(retrieval_direction))

        for scale in scales:
            pred = np.exp(base_log + scale * direction_centered)
            label = str(scale).replace(".", "p").replace("-", "m")
            path = out_dir / f"{args.name_prefix}_s{label}.csv"
            write_submission(base.index, pred, path)

        written_count = len(scales)

    # Summary
    print(f"\nWrote {written_count} submissions to {out_dir}/")
    print("Files:")
    for p in sorted(out_dir.glob("*.csv")):
        print(f"  {p.name}")

    # Recommendation
    print(f"\nRecommended submit order:")
    if orthogonalized:
        print(f"  1. {args.name_prefix}_s0p5_orth.csv  (conservative orth)")
        print(f"  2. {args.name_prefix}_s1p0_orth.csv")
        print(f"  3. {args.name_prefix}_s0p5.csv       (raw direction, if orth works)")
    else:
        print(f"  1. {args.name_prefix}_s0p25.csv  (most conservative)")
        print(f"  2. {args.name_prefix}_s0p5.csv")
        print(f"  3. {args.name_prefix}_s1p0.csv")


if __name__ == "__main__":
    main()
