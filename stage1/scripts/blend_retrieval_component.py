"""
Blend retrieval residual component with the best model's proto/token/torch components.

Usage:
  python scripts/blend_retrieval_component.py \
    --base submission_base.csv \
    --retrieval-residual retrieval_residual_s0p04.csv \
    --best-submission submissions/candidate_336_...csv \
    --name-prefix candidate_retrieval_blend \
    --scales "0.5,1.0,1.5,2.0"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from video_task6b_residual_lab import aligned_values, read_submission, write_submission


def mape(y_true, y_pred):
    y_true = np.maximum(np.abs(y_true), 1e-8)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Blend retrieval residual into component space."
    )
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--retrieval-residual", required=True)
    parser.add_argument("--best-submission")
    parser.add_argument("--scales", default="0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--name-prefix", default="candidate_retrieval_blend")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = read_submission(args.base_submission)
    retrieval_sub = read_submission(args.retrieval_residual)

    aligned = aligned_values([base, retrieval_sub])
    base, retrieval_sub = aligned[0], aligned[1]

    # Retrieval residual in log space
    retrieval_log = np.log(
        np.clip(retrieval_sub.to_numpy(float), 1e-6, None)
        / np.clip(base.to_numpy(float), 1e-6, None)
    )

    # Center the residual
    retrieval_log = retrieval_log - float(np.mean(retrieval_log))

    stats = {
        "retrieval_log_mean": float(np.mean(retrieval_log)),
        "retrieval_log_std": float(np.std(retrieval_log)),
        "retrieval_log_mad": float(np.mean(np.abs(retrieval_log))),
        "retrieval_log_min": float(np.min(retrieval_log)),
        "retrieval_log_max": float(np.max(retrieval_log)),
    }
    print("Retrieval log-residual stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = parse_float_list(args.scales)
    written = []
    for scale in scales:
        pred = base.to_numpy(float) * np.exp(scale * retrieval_log)
        label = str(scale).replace(".", "p").replace("-", "m")
        path = out_dir / f"{args.name_prefix}_s{label}.csv"
        write_submission(base.index, pred, path)
        written.append(str(path))
        print(f"  {path}  scale={scale}")

    # If best submission provided, compute cosine similarity of residual vectors
    if args.best_submission:
        best = read_submission(args.best_submission)
        best = best.loc[base.index]
        best_log = np.log(
            np.clip(best.to_numpy(float), 1e-6, None)
            / np.clip(base.to_numpy(float), 1e-6, None)
        )

        # Cosine similarity between retrieval residual and best residual
        r = retrieval_log - float(np.mean(retrieval_log))
        b = best_log - float(np.mean(best_log))
        cos_sim = float(np.dot(r, b) / (np.sqrt(np.dot(r, r) * np.dot(b, b)) + 1e-12))
        print(f"\n  Cosine similarity (retrieval vs best residual): {cos_sim:.6f}")

        stats["cosine_sim_vs_best"] = cos_sim

    meta_path = out_dir / f"{args.name_prefix}_meta.json"
    meta_path.write_text(json.dumps({**stats, "scales": scales, "written": written}, indent=2))
    print(f"\n  Meta: {meta_path}")
    print(f"  Wrote {len(written)} blended submissions.")


if __name__ == "__main__":
    main()
