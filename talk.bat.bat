@echo off
:: TALK TO DAKAR - No ports, no config, just works

echo ========================================
echo    TALKING TO DAKAR
echo    Type your message and press Enter
echo    Type 'exit' to quit
echo ========================================

:loop
set /p msg="You: "

if "%msg%"=="exit" goto end
if "%msg%"=="" goto loop

:: Send message via file (simplest, always works)
echo %msg% > C:\NEXUS\talk_to_dakar.txt

:: Wait for response (Dakar watches this file)
echo Waiting for Dakar...
timeout /t 2 /nobreak > nul

:: Show response if any
if exist C:\NEXUS\dakar_response.txt (
    type C:\NEXUS\dakar_response.txt
    del C:\NEXUS\dakar_response.txt
) else (
    echo Dakar is thinking...
)

goto loop

:end
echo Goodbye.