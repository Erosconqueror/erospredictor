@echo off
title Erospredictor DEV collector - Iron to Challenger
call "%~dp0scrape_continuous.bat" --key-profile development --tiers IRON BRONZE SILVER GOLD PLATINUM EMERALD DIAMOND MASTER GRANDMASTER CHALLENGER %*
exit /b %ERRORLEVEL%
