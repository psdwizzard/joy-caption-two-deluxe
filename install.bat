@echo off
SETLOCAL EnableDelayedExpansion

echo ===================================================
echo Image Captioning Application - Installation Script
echo ===================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Create virtual environment directory if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install Pillow
pip install PyQt5

REM Create launcher script
echo Creating launcher script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo python dark_mode_gui.py
echo pause
) > run_app.bat

echo.
echo ===================================================
echo Installation complete!
echo ===================================================
echo.
echo To run the application, double-click on run_app.bat
echo.
pause