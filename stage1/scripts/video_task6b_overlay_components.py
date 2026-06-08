from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from video_task6b_residual_lab import aligned_values, read_submission, write_submission


def parse_component(text: str) -> tuple[Path, float, bool, Path | None]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in {2, 3, 4}:
        raise ValueError("--component must be path,weight[,center[,anchor_path]]")
    path = Path(parts[0])
    weight = float(parts[1])
    center = len(parts) == 3 and parts[2].lower() in {"1", "true", "yes", "center", "centered"}
    if len(parts) == 4:
        center = parts[2].lower() in {"1", "true", "yes", "center", "centered"}
    anchor = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    return path, weight, center, anchor


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend SMP video submissions in log residual space.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--component", action="append", required=True, help="path,weight[,center[,anchor_path]]")
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/overlay_components_report.json")
    args = parser.parse_args()

    base = read_submission(args.base)
    component_defs = [parse_component(x) for x in args.component]
    component_series = [read_submission(path) for path, _weight, _center, _anchor in component_defs]
    anchor_series = [read_submission(anchor) for _path, _weight, _center, anchor in component_defs if anchor is not None]
    aligned = aligned_values([base] + component_series + anchor_series)
    base = aligned[0]
    component_series = aligned[1 : 1 + len(component_defs)]
    aligned_anchor_series = aligned[1 + len(component_defs) :]
    anchor_iter = iter(aligned_anchor_series)

    eps = 1e-6
    base_log = np.log(np.clip(base.to_numpy(float), eps, None))
    total = np.zeros_like(base_log)
    rows = []
    for (path, weight, center, anchor), series in zip(component_defs, component_series):
        if anchor is None:
            anchor_log = base_log
        else:
            anchor_values = next(anchor_iter)
            anchor_log = np.log(np.clip(anchor_values.to_numpy(float), eps, None))
        vec = np.log(np.clip(series.to_numpy(float), eps, None)) - anchor_log
        raw = vec.copy()
        if center:
            vec = vec - float(np.mean(vec))
        total = total + weight * vec
        rows.append(
            {
                "path": str(path),
                "weight": weight,
                "center": center,
                "anchor": str(anchor) if anchor is not None else args.base,
                "raw": vector_stats(raw),
                "used": vector_stats(vec),
            }
        )

    pred = np.exp(base_log + total)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.name_prefix}.csv"
    write_submission(base.index, pred, output)

    delta = pred - base.to_numpy(float)
    report = {
        "base": args.base,
        "output": str(output),
        "components": rows,
        "total_log_vector": vector_stats(total),
        "delta_vs_base": vector_stats(delta),
        "mean_pred": float(np.mean(pred)),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
