@echo off
title RoadSafe - Quickstart Demo
cd /d "%~dp0"
echo ====================================================================
echo  RoadSafe: Road Damage Detection and Severity Assessment Demo
echo ====================================================================
echo.
python demo.py
echo.
echo ====================================================================
echo  Demo finished! Press any key to close this window.
echo ====================================================================
pause >nul
