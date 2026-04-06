@echo off
color F
call env\Scripts\activate.bat
cd erospredictor
python updater.py
echo Frissitesi procesure veget ert...
color A
echo Erospredictor inditasa...
python erospredictor.py
color C
echo Erospredictor leallitva!
pause