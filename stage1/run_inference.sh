#!/bin/bash
# ============================================================
# One-click Inference Script
# Reproduces best submission MAPE 0.17018 exactly.
# Requires: Python environment from environment.yml
# ============================================================
set -e

echo "============================================"
echo "SMP 2026 Video Task 6b — Inference"
echo "============================================"

# Download feature files if not present (replace with your storage URL)
FEATURE_DIR="features"
if [ ! -f "$FEATURE_DIR/video_asr.csv" ]; then
    echo "[NOTE] Large feature files not found."
    echo "  Exact reproduction uses only component CSVs (already included)."
    echo "  Feature files are only needed for re-training from scratch."
    echo "  To train: download feature files and run run_training.sh"
    echo ""
fi

# Run exact reproduction
echo "[1/1] Reproducing best submission from component CSVs..."
cd scripts
python reproduce_best_from_components.py
cd ..

# Verify
DIFF=$(python -c "
import pandas as pd, numpy as np
a = pd.read_csv('submissions/candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv')
b = pd.read_csv('reproduced/candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv')
print(np.max(np.abs(a['polularity_score'].values - b['polularity_score'].values)))
")

echo ""
echo "============================================"
echo "Max absolute difference: $DIFF"
if python -c "exit(0 if float('$DIFF') < 1e-10 else 1)"; then
    echo "REPRODUCTION VERIFIED — Output matches submission exactly."
    echo "Final file: reproduced/candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv"
else
    echo "WARNING: Mismatch detected."
fi
echo "============================================"
