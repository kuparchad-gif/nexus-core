@echo off
echo ===================================================
echo       VIREN DOCUMENT SUITE LAUNCHER
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Create required directories
if not exist uploads mkdir uploads
if not exist documents mkdir documents
if not exist logs mkdir logs

echo [INFO] Checking dependencies...
python -c "import gradio, pandas, pillow" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required dependencies...
    pip install gradio pandas pillow python-docx PyPDF2 openpyxl autopep8 qrcode
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo [INFO] Launching Viren Document Suite...
python viren_document_suite.py

echo [SUCCESS] Viren Document Suite launched successfully!
echo The interface will open in your web browser shortly.
echo If it doesn't open automatically, navigate to http://localhost:7860
echo.
echo Press any key to close this window...
pause > nul