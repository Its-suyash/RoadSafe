@echo off
title RoadSafe - Environment & Dependency Setup
cd /d "%~dp0"
echo ====================================================================
echo  RoadSafe: Installing Required Python Packages
echo ====================================================================
echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Standard pip install encountered an issue. Trying with user flag...
    pip install --user -r requirements.txt
)

echo.
echo ====================================================================
echo  Setup Complete! You can now double-click 1_RUN_DEMO.bat
echo ====================================================================
pause
