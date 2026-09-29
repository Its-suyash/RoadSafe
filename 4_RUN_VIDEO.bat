@echo off
title RoadSafe - Video / Webcam Inference
cd /d "%~dp0"
echo ====================================================================
echo  RoadSafe: Road Damage Detection - Video / Webcam Mode
echo ====================================================================
echo.
echo  Choose a mode:
echo    [1] Run on built-in sample road video
echo    [2] Enter path to your own video file
echo    [3] Run live detection on webcam
echo.
set /p choice="  Enter choice (1, 2, or 3): "
echo.

if "%choice%"=="1" (
    echo  Running detection on sample video: demo video\mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4
    echo  (Press 'q' in the preview window anytime to finish early)
    python inference.py --video "demo video\mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4" --output results/ --device cpu
) else if "%choice%"=="2" (
    set /p videopath="  Enter path to video file: "
    echo.
    echo  Running detection on video...
    python inference.py --video "%videopath%" --output results/ --device cpu
) else if "%choice%"=="3" (
    echo  Starting webcam detection (press 'q' in the window to stop)...
    python inference.py --webcam 0 --output results/ --device cpu
) else (
    echo  Invalid choice. Please run again and enter 1, 2, or 3.
)

echo.
echo ====================================================================
echo  Done! Press any key to close this window.
echo ====================================================================
pause >nul
