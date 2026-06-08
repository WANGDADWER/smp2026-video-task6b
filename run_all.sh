#!/bin/bash
# ============================================================
# SMP 2026 Video Task 6b — Complete 2-Stage Pipeline
# ============================================================
# Stage 1: Exact reproduction of candidate_336 (no GPU needed)
# Stage 2: MLP [256,128,64] + centered residual blend (GPU recommended)
#
# Output: submissions/final_submission.csv
# ============================================================
set -e

echo "============================================"
echo "SMP 2026 Video Task 6b — Full Pipeline"
echo "============================================"
echo ""

# ============================================
# Stage 1: Reproduce candidate_336
# ============================================
echo "[Stage 1/2] Reproducing candidate_336..."
echo "----------------------------------------"

cd stage1

# Check if candidate_336 already exists
if [ -f "submissions/candidate_336_pf_after332_2_protop1p345_tokenp0p970_torchp1p000.csv" ]; then
    echo "  candidate_336 already exists. Skipping reproduction."
    echo "  To force re-run: rm stage1/submissions/candidate_336_*.csv && bash run_all.sh"
else
    echo "  Reproducing from component CSVs..."
    cd scripts
    python reproduce_best_from_components.py
    cd ..
    echo "  Done."
fi

cd ..

# ============================================
# Stage 2: MLP + Centered Residual Blend
# ============================================
echo ""
echo "[Stage 2/2] MLP + candidate_336 centered residual blend..."
echo "----------------------------------------"

cd stage2
python run_train.py --predict-only --alpha 0.03
cd ..

# Copy final submission
cp submissions/stage2_centered_a0.030.csv submissions/final_submission.csv

echo ""
echo "============================================"
echo "Pipeline complete!"
echo ""
echo "Final submission: submissions/final_submission.csv"
echo ""
echo "Files generated:"
echo "  Stage 1:"
echo "    stage1/submissions/candidate_336_*.csv"
echo "  Stage 2:"
echo "    submissions/stage2_mlp_base.csv"
echo "    submissions/stage2_centered_a*.csv"
echo "    submissions/final_submission.csv  <-- FINAL"
echo "============================================"
