:: C:\Engineers\eden_engineering\scripts\model_selector.bat
:: Double-click to launch the model selector in your environment

@echo off
cd /d %~dp0
poetry run python scripts\model_selector.py
pause
