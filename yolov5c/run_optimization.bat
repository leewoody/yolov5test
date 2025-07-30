@echo off
echo Starting Regurgitation Dataset Optimization...

REM Run automated optimization for Regurgitation dataset
python3 auto_optimize_regurgitation.py ^
    --dataset ../Regurgitation-YOLODataset-1new ^
    --epochs 100 ^
    --batch-size 16

echo Optimization completed!
pause 