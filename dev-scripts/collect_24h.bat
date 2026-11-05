@echo off
setlocal
title Erospredictor - 24 hour collection
"%~dp0..\env\Scripts\python.exe" -u "%~dp0..\erospredictor\collection_day.py" %*
set "DAY_EXIT=%ERRORLEVEL%"
pause
exit /b %DAY_EXIT%
