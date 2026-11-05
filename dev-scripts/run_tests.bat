@echo off
setlocal
cd /d "%~dp0..\erospredictor"
set "QT_QPA_PLATFORM=offscreen"
"%~dp0..\env\Scripts\python.exe" -m unittest discover -s tests -t . %*
set "TEST_EXIT=%ERRORLEVEL%"
pause
exit /b %TEST_EXIT%
