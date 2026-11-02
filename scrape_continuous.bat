@echo off
rem Double-click: collect in rank-profile rounds without training or replacing models.
call "%~dp0update_models.bat" --scrape-only %*
exit /b %ERRORLEVEL%
