from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = PACKAGE_ROOT / "components"
SUBMISSION_DIR = PACKAGE_ROOT / "submissions"
OUTPUT_DIR = PACKAGE_ROOT / "reproduced"


BEST_NAME = "candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv"
TOKEN_0975_NAME = "candidate_332_pf_component_updated_2_protop1p345_tokenp0p975_torchp1p000.csv"
TOKEN_0950_NAME = "candidate_332_pf_component_updated_5_protop1p345_tokenp0p950_torchp1p000.csv"


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["pid", "polularity_score"]
    if list(df.columns) != expected:
        raise ValueError(f"{path} columns are {list(df.columns)}, expected {expected}")
    if len(df) != 2000:
        raise ValueError(f"{path} has {len(df)} rows, expected 2000")
    return df


def interpolate_token(
    token_0975: pd.DataFrame,
    token_0950: pd.DataFrame,
    target_token: float,
) -> pd.DataFrame:
    if token_0975["pid"].astype(str).tolist() != token_0950["pid"].astype(str).tolist():
        raise ValueError("PID order mismatch between component submissions")

    lo = 0.950
    hi = 0.975
    alpha = (target_token - hi) / (lo - hi)
    pred = token_0975["polularity_score"].to_numpy(dtype=float)
    direction = token_0950["polularity_score"].to_numpy(dtype=float) - pred
    out = token_0975.copy()
    out["polularity_score"] = np.clip(pred + alpha * direction, 1e-6, None)
    return out


def write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if a["pid"].astype(str).tolist() != b["pid"].astype(str).tolist():
        raise ValueError("PID order mismatch while comparing submissions")
    return float(
        np.max(
            np.abs(
                a["polularity_score"].to_numpy(dtype=float)
                - b["polularity_score"].to_numpy(dtype=float)
            )
        )
    )


def main() -> None:
    token_0975 = read_submission(COMPONENT_DIR / TOKEN_0975_NAME)
    token_0950 = read_submission(COMPONENT_DIR / TOKEN_0950_NAME)
    included_best = read_submission(SUBMISSION_DIR / BEST_NAME)

    reproduced_best = interpolate_token(token_0975, token_0950, target_token=0.970)
    best_path = OUTPUT_DIR / BEST_NAME
    write_submission(reproduced_best, best_path)

    diff = max_abs_diff(reproduced_best, included_best)
    print(f"Reproduced best: {best_path}")
    print(f"Max absolute difference vs included best: {diff:.3e}")
    if diff > 1e-10:
        raise SystemExit("Reproduction check failed")

    probes = {
        "candidate_340_pf_after336_micro_4_protop1p3450_tokenp0p96950_torchp1p000.csv": 0.96950,
        "candidate_340_pf_after336_micro_1_protop1p3450_tokenp0p96875_torchp1p000.csv": 0.96875,
        "candidate_339_pf_after336_fine_1_protop1p3450_tokenp0p9675_torchp1p000.csv": 0.96750,
        "candidate_340_pf_after336_micro_2_protop1p3450_tokenp0p96625_torchp1p000.csv": 0.96625,
        "candidate_336_pf_after332_4_protop1p345_tokenp0p965_torchp1p000.csv": 0.96500,
    }
    for name, target_token in probes.items():
        write_submission(
            interpolate_token(token_0975, token_0950, target_token=target_token),
            OUTPUT_DIR / name,
        )

    print(f"Wrote {1 + len(probes)} submissions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

