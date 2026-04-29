@echo off
color 0B
echo [1/3] Python 3.11 ellenorzese a gepen...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python 3.11 nem talalhato, letoltes es telepites a hatterben...
    curl -L -o python_installer.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_test=0
    del python_installer.exe
    echo [OK] Python 3.11 telepitve
) else (
    echo [OK] Python 3.11 mar telepitve van
)

echo.
echo Erospredictor program elofelteteleinek telepitese:
echo [2/3] Virtualis kornyezet letrehozasa es aktivalasa...
py -3.11 -m venv env
call env\Scripts\activate.bat

echo [3/3] Szukseges python csomagok telepitese...
python -m pip install --upgrade pip
pip install -r erospredictor\data\requirements.txt

echo Elofelteleket telepitve, a program futtathato
pause