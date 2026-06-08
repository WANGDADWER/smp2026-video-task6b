"""
Optimize residual blending weights to fit original component CSVs.

Method:
  1. For each component CSV (comp_2, comp_5, comp_7), grid-search
     proto/token/torch weights to minimize MAPE between our blend
     and the target component CSV:
       blend = base × exp(w1×proto_resid + w2×token_resid + w3×torch_resid)

  2. Linear interpolation between fitted components to match best CSV:
       final = comp_i + alpha × (comp_j - comp_i)

Output: optimized blend CSV + weight report JSON.
"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import pandas as pd


def load_log_resid(files, base_pids, log_b):
    """Load log-space residuals: log(component) - log(base)."""
    result = {}
    for f in files:
        name = Path(f).stem
        df = pd.read_csv(f).drop_duplicates("pid")
        vals = df.set_index("pid")["polularity_score"]
        vals = np.clip(vals.loc[base_pids].values, 1e-6, None)
        result[name] = np.log(vals) - log_b
    return result


def search_weights(log_b, residual_dicts, target):
    """Grid-search best 3-way weights to match target."""
    best_mape = 999
    best_info = None
    proto_dict, token_dict, torch_dict = residual_dicts
    for pk, p_lr in proto_dict.items():
        for tk, t_lr in token_dict.items():
            for trk, tr_lr in torch_dict.items():
                for w1 in np.arange(0.3, 2.5, 0.05):
                    for w2 in np.arange(0.05, 2.0, 0.05):
                        for w3 in np.arange(0.05, 1.5, 0.05):
                            pred = np.exp(log_b + w1*p_lr + w2*t_lr + w3*tr_lr)
                            m = float(np.mean(np.abs(pred - target) / target))
                            if m < best_mape:
                                best_mape = m
                                best_info = (pk, tk, trk, w1, w2, w3, pred.copy())
    return best_info, best_mape


def main():
    parser = argparse.ArgumentParser(description="Optimize residual blending weights")
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--residual-dir", required=True)
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--best-submission", required=True)
    parser.add_argument("--output-dir", default="outputs/video_task6b/submissions")
    parser.add_argument("--report", default="outputs/video_task6b/reports/optimize_weights_report.json")
    args = parser.parse_args()

    # Load base
    base = pd.read_csv(args.base_submission).drop_duplicates("pid")
    b = base.set_index("pid")["polularity_score"]
    pids = b.index
    b_vals = b.loc[pids].values
    log_b = np.log(np.clip(b_vals, 1e-6, None))

    # Load targets
    comp_dir = Path(args.component_dir)
    comp_2 = pd.read_csv(comp_dir / "candidate_332_pf_component_updated_2_protop1p345_tokenp0p975_torchp1p000.csv")
    comp_5 = pd.read_csv(comp_dir / "candidate_332_pf_component_updated_5_protop1p345_tokenp0p950_torchp1p000.csv")
    comp_7 = pd.read_csv(comp_dir / "candidate_332_pf_component_updated_7_protop1p350_tokenp0p950_torchp1p000.csv")
    best = pd.read_csv(args.best_submission)

    targets = {
        "comp_2": comp_2.set_index("pid")["polularity_score"].loc[pids].values,
        "comp_5": comp_5.set_index("pid")["polularity_score"].loc[pids].values,
        "comp_7": comp_7.set_index("pid")["polularity_score"].loc[pids].values,
        "best": best.set_index("pid")["polularity_score"].loc[pids].values,
    }

    # Load residuals
    res_dir = Path(args.residual_dir)
    proto_files = sorted(res_dir.glob("candidate_250_proto_resid_s*.csv"))
    token_files = sorted(res_dir.glob("candidate_273_token_trend_*_s0p05.csv"))
    torch_files = sorted(res_dir.glob("candidate_278_torch_fusion_s*.csv"))

    if not proto_files or not token_files or not torch_files:
        print("ERROR: Residual files not found. Run residual training first.")
        return

    proto_dict = load_log_resid([str(f) for f in proto_files], pids, log_b)
    token_dict = load_log_resid([str(f) for f in token_files], pids, log_b)
    torch_dict = load_log_resid([str(f) for f in torch_files], pids, log_b)
    residual_dicts = (proto_dict, token_dict, torch_dict)

    # Step 1: Fit each component CSV
    print("=== Step 1: Fit each component CSV ===")
    fitted = {}
    for comp_name in ["comp_2", "comp_5", "comp_7"]:
        (pk, tk, trk, w1, w2, w3, pred), mape = search_weights(
            log_b, residual_dicts, targets[comp_name])
        fitted[comp_name] = {"proto_file": pk, "token_file": tk, "torch_file": trk,
                              "proto_w": w1, "token_w": w2, "torch_w": w3, "pred": pred}
        print(f"  {comp_name}: weights=({w1:.2f},{w2:.2f},{w3:.2f}) MAPE={mape*100:.3f}%")

    # Step 2: Linear interpolation to best
    print("\n=== Step 2: Linear interpolation to best ===")
    best_mape = 999
    best_info = None
    for a, na in [("comp_2","comp_5"),("comp_2","comp_7"),("comp_5","comp_7")]:
        y1 = fitted[a]["pred"]
        y2 = fitted[na]["pred"]
        for alpha in np.arange(-1.5, 3.0, 0.01):
            interp = y1 + alpha * (y2 - y1)
            m = float(np.mean(np.abs(interp - targets["best"]) / targets["best"]))
            if m < best_mape:
                best_mape = m
                best_info = (a, na, alpha, interp.copy())

    a, na, alpha, final_pred = best_info
    print(f"  Best: {a} + {alpha:.2f}×({na} - {a})")
    print(f"  MAPE vs best: {best_mape*100:.4f}%")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidate_optimized_blend.csv"
    out = pd.DataFrame({"pid": pids, "polularity_score": np.clip(final_pred, 1e-6, None)})
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Report
    report = {
        "fitted_components": {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                              for k, v in fitted.items()},
        "interpolation": {"anchor": a, "source": na, "alpha": alpha},
        "mape_vs_best": best_mape,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
