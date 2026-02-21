@echo off

echo ======================================
echo   Setting up PostVTON (Conda Windows)
echo ======================================

REM -------- CONFIG --------

set ENV_NAME=vton
set PYTHON_VERSION=3.10

set CATVTON_REPO=https://github.com/Zheng-Chong/CatVTON.git
set OOTD_REPO=https://github.com/levihsu/OOTDiffusion.git

set CATVTON_DIR=external\catvton
set OOTD_DIR=external\ootdiffusion

REM ------------------------

REM Activate conda base
call conda activate base

if errorlevel 1 (
    echo Conda not found. Please run in Anaconda Prompt.
    pause
    exit /b 1
)

echo Conda found

REM Create env
conda env list | findstr %ENV_NAME% >nul
if %errorlevel%==0 (
    echo Env %ENV_NAME% already exists
) else (
    echo Creating env %ENV_NAME%...
    conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
)

REM Activate env
call conda activate %ENV_NAME%

echo Activated %ENV_NAME%

REM Install PyTorch (CUDA 13.0)
call conda install -y pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

REM Upgrade pip
python -m pip install --upgrade pip

REM Install main deps
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found
)

REM Create external folder
if not exist external mkdir external

REM Clone CatVTON
if not exist %CATVTON_DIR% (
    git clone %CATVTON_REPO% %CATVTON_DIR%
) else (
    echo CatVTON exists
)

REM Install CatVTON deps
if exist %CATVTON_DIR%\requirements.txt (
    pip install -r %CATVTON_DIR%\requirements.txt
)

REM Clone OOTDiffusion
if not exist %OOTD_DIR% (
    git clone %OOTD_REPO% %OOTD_DIR%
) else (
    echo OOTDiffusion exists
)

REM Install OOTDiffusion deps
if exist %OOTD_DIR%\requirements.txt (
    pip install -r %OOTD_DIR%\requirements.txt
)

REM GPU check
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo GPU detected
    nvidia-smi | more
) else (
    echo No GPU detected
)

echo ======================================
echo Setup Complete!
echo Activate later with:
echo   conda activate %ENV_NAME%
echo ======================================

pause
