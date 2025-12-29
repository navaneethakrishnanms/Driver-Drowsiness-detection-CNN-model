@echo off
echo ================================================
echo Driver Drowsiness Detection System
echo ================================================
echo.
cd /d "%~dp0scripts"
python driver_drowsiness_cnn.py
pause
