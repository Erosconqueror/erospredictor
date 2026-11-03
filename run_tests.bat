@echo off
setlocal
cd /d "%~dp0erospredictor"
set "QT_QPA_PLATFORM=offscreen"
"%~dp0env\Scripts\python.exe" -m unittest discover -s tests -t . %*
set "TEST_EXIT=%ERRORLEVEL%"
pause
exit /b %TEST_EXIT%
