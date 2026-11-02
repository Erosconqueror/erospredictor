@echo off
rem Double-click: rebuild statistics, roles, blind profiles and all models from stored matches.
call "%~dp0update_models.bat" --skip-scrape %*
exit /b %ERRORLEVEL%
