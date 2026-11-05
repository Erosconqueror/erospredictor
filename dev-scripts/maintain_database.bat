@echo off
setlocal
if not exist "%~dp0..\env\Scripts\python.exe" (
    echo A projekt Python kornyezete hianyzik. Eloszor futtasd az install.bat fajlt.
    pause
    exit /b 1
)
echo Fejlesztoi adatbazis-karbantartas. Alapbol csak elonezet; torleshez --apply.
"%~dp0..\env\Scripts\python.exe" "%~dp0..\erospredictor\database_maintenance.py" %*
set "MAINTENANCE_EXIT=%ERRORLEVEL%"
pause
exit /b %MAINTENANCE_EXIT%
