@echo off
setlocal
pushd "%~dp0"
color 0B
echo [1/4] Python 3.11 ellenorzese a gepen...
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
echo [2/4] Virtualis kornyezet letrehozasa es aktivalasa...
py -3.11 -m venv env
call env\Scripts\activate.bat

echo [3/4] Szukseges python csomagok telepitese...
python -m pip install --upgrade pip
pip install -r erospredictor\data\requirements.txt
if errorlevel 1 (
    echo A csomagok telepitese nem sikerult. Javitsd a hibat, majd futtasd ujra a telepitot.
    popd
    pause
    exit /b 1
)

echo [4/4] A helyi League of Legends kliens megkeresese es utvonalanak mentese...
python erospredictor\configure_client.py

echo Elofelteleket telepitve, a program futtathato
popd
pause
