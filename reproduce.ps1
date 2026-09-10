# Quick Evaluation Reproduction Script (PowerShell)
# Reproduces the entire Hiver Support Agent evaluation benchmark in < 15 minutes.
# Uses precomputed offline artifacts with zero API spend.

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "=== HIVER SDE INTERN: 15-MINUTE EVALUATION REPRODUCTION ===" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$StartTime = Get-Date

# 1. Verify Virtual Environment
If (Test-Path ".venv\Scripts\python.exe") {
    $PYTHON = ".venv\Scripts\python.exe"
} Else {
    $PYTHON = "python"
}

Write-Host "`n[1/3] Running Full Modular Test Suite (9 Test Files, 38 Tests)..." -ForegroundColor Yellow
& $PYTHON -m pytest tests/ -v
If ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pipeline tests failed. Aborting." -ForegroundColor Red
    Exit 1
}
Write-Host "[PASS] All unit and integration tests passed!" -ForegroundColor Green

# 2. Run Comprehensive Evaluation Harness
Write-Host "`n[2/3] Running 3-System Evaluation Harness on 200 Golden Records..." -ForegroundColor Yellow
& $PYTHON -u eval/run_eval.py --use-cache --limit 200
If ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Evaluation harness failed. Aborting." -ForegroundColor Red
    Exit 1
}

# 3. Print Results Summary
Write-Host "`n[3/3] Displaying Final Baseline Comparison Table:" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
Get-Content artifacts/baseline_comparison_table.md
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray

$EndTime = Get-Date
$Elapsed = [math]::Round(($EndTime - $StartTime).TotalSeconds, 2)

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Complete evaluation reproduced in $Elapsed seconds!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
