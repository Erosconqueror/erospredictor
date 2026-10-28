@echo off
setlocal
if not exist "%~dp0env\Scripts\python.exe" (
    echo A projekt Python kornyezete hianyzik. Eloszor futtasd az install.bat fajlt.
    pause
    exit /b 1
)
"%~dp0env\Scripts\python.exe" "%~dp0erospredictor\update_pipeline.py" %*
set "UPDATE_EXIT=%ERRORLEVEL%"
if not "%UPDATE_EXIT%"=="0" echo A frissites nem fejezodott be. A korabbi modellek megmaradtak.
pause
exit /b %UPDATE_EXIT%
