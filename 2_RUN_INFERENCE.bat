@echo off
title RoadSafe - Batch Inference Pipeline
cd /d "%~dp0"
echo ====================================================================
echo  RoadSafe: Running Full Batch Inference on Test Split
echo ====================================================================
if exist "dataset\test\images" (
    echo  Running inference on full test dataset (dataset\test\images)...
    python inference.py --image-dir dataset/test/images/ --output results/
) else (
    echo  Running inference on sample images (sample_images/)...
    python inference.py --image-dir sample_images/ --output results/
)
echo ====================================================================
echo  Inference Complete! Annotated images and results.json saved to results/
echo ====================================================================
pause
