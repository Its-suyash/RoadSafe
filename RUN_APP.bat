@echo off
title RoadSafe - Web Application
cd /d "%~dp0"
echo ====================================================================
echo  RoadSafe: Road Damage & Severity Assessment - Web App
echo ====================================================================
echo.
echo  Starting RoadSafe Web App...
echo  Your browser will open automatically at http://localhost:8501
echo  (Press Ctrl+C in this window anytime to stop the app)
echo.
streamlit run app.py
pause
