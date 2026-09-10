#!/usr/bin/env bash
# Quick Evaluation Reproduction Script (Bash)
# Reproduces the entire Hiver Support Agent evaluation benchmark in < 15 minutes.
# Uses precomputed offline artifacts with zero API spend.

set -e

echo "============================================================"
echo "=== HIVER SDE INTERN: 15-MINUTE EVALUATION REPRODUCTION ==="
echo "============================================================"

START_TIME=$(date +%s)

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="python"
fi

echo ""
echo "[1/3] Running Full Modular Test Suite (9 Test Files, 38 Tests)..."
$PYTHON -m pytest tests/ -v
echo "[PASS] All unit and integration tests passed!"

echo ""
echo "[2/3] Running 3-System Evaluation Harness on 200 Golden Records..."
$PYTHON -u eval/run_eval.py --use-cache --limit 200

echo ""
echo "[3/3] Displaying Final Baseline Comparison Table:"
echo "------------------------------------------------------------"
cat artifacts/baseline_comparison_table.md
echo "------------------------------------------------------------"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "[SUCCESS] Complete evaluation reproduced in ${ELAPSED} seconds!"
echo "============================================================"
