@echo off
setlocal
if not exist "%~dp0env\Scripts\python.exe" (
    echo A projekt Python kornyezete hianyzik. Eloszor futtasd az install.bat fajlt.
    pause
    exit /b 1
)
set "UPDATE_RESUME="
:RUN_UPDATE
"%~dp0env\Scripts\python.exe" -u "%~dp0erospredictor\update_pipeline.py" %* %UPDATE_RESUME%
set "UPDATE_EXIT=%ERRORLEVEL%"
if "%UPDATE_EXIT%"=="0" goto UPDATE_DONE
if "%UPDATE_EXIT%"=="130" exit /b 130
if "%UPDATE_EXIT%"=="-1073741510" exit /b 130
if "%UPDATE_EXIT%"=="3221225786" exit /b 130
if "%UPDATE_EXIT%"=="2" goto UPDATE_ATTENTION
echo A frissito folyamata megszakadt. Folytatas a mentett pontrol 10 masodperc mulva. Ctrl+C: leallitas.
timeout /t 10 /nobreak >nul
if errorlevel 1 exit /b 130
set "UPDATE_RESUME=--resume"
goto RUN_UPDATE
:UPDATE_ATTENTION
echo A frissites beavatkozast igenyel. A folytatasi pont es a korabbi modellek megmaradtak.
:UPDATE_DONE
pause
exit /b %UPDATE_EXIT%
