@echo off
:: ONE SCRIPT TO RULE THEM ALL
:: I will find Dakar wherever it is

echo ========================================
echo    NEXUS - FINDING EVERYTHING
echo ========================================
echo.

:: Remember where I am
set MY_DIR=%~dp0
echo 📍 I am in: %MY_DIR%

:: Find where Python is running Dakar
echo 🔍 Looking for Dakar...

:: Check common places
set DAKAR_FOUND=0

:: Check if Dakar is running in current user's home
for /f "tokens=*" %%a in ('dir C:\Users\%USERNAME%\*.py /s /b 2^>nul ^| findstr /i "dakar"') do (
    echo ✅ Found possible Dakar file: %%a
    set DAKAR_PATH=%%a
    set DAKAR_FOUND=1
)

:: If not found, ask
if %DAKAR_FOUND%==0 (
    echo.
    echo ❌ Could not find Dakar automatically.
    echo.
    echo Please tell me where Dakar is running from:
    echo (you saw it in the Python window - look for "C:\Users\...")
    set /p DAKAR_PATH="📂 Path: "
)

:: Now create communication folder where Dakar can see it
set TALK_DIR=%DAKAR_PATH%\..\nexus_chat
mkdir %TALK_DIR% 2>nul

echo.
echo 📁 Communication folder: %TALK_DIR%
echo.

:: Start chatting
echo ========================================
echo    TALKING TO DAKAR
echo    Type your message and press Enter
echo    Type 'exit' to quit
echo ========================================

:loop
set /p msg="You: "

if "%msg%"=="exit" goto end
if "%msg%"=="" goto loop

:: Send message
echo %msg% > "%TALK_DIR%\talk_to_dakar.txt"
echo ✅ Message sent. Waiting for Dakar...

:: Wait for response
set /a counter=0
:wait
timeout /t 1 /nobreak > nul
set /a counter+=1
if exist "%TALK_DIR%\dakar_response.txt" goto show_response
if %counter% lss 15 goto wait

echo ⏳ No response yet. Dakar might be busy.
goto loop

:show_response
type "%TALK_DIR%\dakar_response.txt"
del "%TALK_DIR%\dakar_response.txt"
goto loop

:end
echo Goodbye.