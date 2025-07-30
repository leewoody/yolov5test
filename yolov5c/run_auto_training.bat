@echo off
echo Starting automated training for multiple datasets...

REM Run automated training for multiple datasets
python3 auto_train_multiple_datasets.py ^
    --datasets ../demo ../demo2 ../Regurgitation-YOLODataset-1new ^
    --weights yolov5s.pt ^
    --hyperparameters hyp_improved.yaml ^
    --epochs 3 ^
    --batch-size 4 ^
    --image-size 640 ^
    --project runs/auto_training ^
    --create-detection-only ^
    --timeout 1800

echo Automated training completed!
pause 