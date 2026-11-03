@echo off
setlocal
if not exist "%~dp0env\Scripts\python.exe" (
    echo Run install.bat first.
    pause
    exit /b 1
)
set "COLLECT_RESUME="
:RUN_COLLECTOR
"%~dp0env\Scripts\python.exe" -u "%~dp0erospredictor\collector_script.py" %* %COLLECT_RESUME%
set "COLLECT_EXIT=%ERRORLEVEL%"
if "%COLLECT_EXIT%"=="0" goto COLLECT_DONE
if "%COLLECT_EXIT%"=="2" goto COLLECT_DONE
if "%COLLECT_EXIT%"=="130" exit /b 130
if "%COLLECT_EXIT%"=="-1073741510" exit /b 130
if "%COLLECT_EXIT%"=="3221225786" exit /b 130
echo Collector interrupted. Resume in 10 seconds. Ctrl+C: stop.
timeout /t 10 /nobreak >nul
if errorlevel 1 exit /b 130
set "COLLECT_RESUME=--resume"
goto RUN_COLLECTOR
:COLLECT_DONE
pause
exit /b %COLLECT_EXIT%
