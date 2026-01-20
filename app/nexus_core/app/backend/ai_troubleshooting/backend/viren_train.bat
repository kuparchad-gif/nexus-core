@echo off
title VIREN EVOLUTION SYSTEM - ONE CLICK LAUNCHER
color 0A

:menu
cls
echo.
echo ========================================
echo    VIREN EVOLUTION SYSTEM - LAUNCHER
echo ========================================
echo.
echo  1. QUICK TRAIN: Neural Networks (Turbo)
echo  2. QUICK TRAIN: Transformers (Hyper)
echo  3. BURST TRAIN: Maximum Power Mode
echo  4. EXIT
echo.
set /p choice="SELECT OPTION (1-4): "

if "%choice%"=="1" goto neural
if "%choice%"=="2" goto transformer  
if "%choice%"=="3" goto burst
if "%choice%"=="4" exit

echo INVALID CHOICE - Press any key to retry
pause >nul
goto menu

:neural
echo.
echo LAUNCHING: Neural Networks Turbo Training...
python -c "from viren_evolution_system import quick_train; quick_train('neural_networks', 'turbo', 'standard', 'auto')"
echo.
echo TRAINING COMPLETE - Press any key to continue
pause >nul
goto menu

:transformer
echo.
echo LAUNCHING: Transformer Hyper Training...
python -c "from viren_evolution_system import quick_train; quick_train('transformer_models', 'hyper', 'comprehensive', 'auto')"
echo.
echo TRAINING COMPLETE - Press any key to continue  
pause >nul
goto menu

:burst
echo.
echo LAUNCHING: BURST MODE - Maximum Power...
python -c "from viren_evolution_system import burst_train; burst_train('all_topics', True)"
echo.
echo BURST TRAINING STARTED - Press any key to continue
pause >nul
goto menu