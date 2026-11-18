@echo off
cd /d "%~dp0..\erospredictor"
"%~dp0..\env\Scripts\python.exe" export_champion_traits.py %*
pause
