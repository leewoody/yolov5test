@echo off
REM Improved YOLOv5 training script with fixed classification loss handling

echo Starting improved YOLOv5 training...

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;.

REM Run training with improved settings
python train_improved.py ^
    --data data.yaml ^
    --weights yolov5s.pt ^
    --hyp hyp_improved.yaml ^
    --epochs 100 ^
    --batch-size 16 ^
    --imgsz 640 ^
    --project runs/train ^
    --name improved_training ^
    --exist-ok

echo Training completed!
pause
