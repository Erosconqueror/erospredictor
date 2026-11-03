@echo off
rem Double-click: rebuild statistics, roles, blind profiles and all models from stored matches.
call "%~dp0update_models.bat" --skip-scrape --all-regions %*
exit /b %ERRORLEVEL%
