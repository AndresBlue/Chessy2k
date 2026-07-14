@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Prefer a local .venv; otherwise Python 3.10 via the py launcher.
set "PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3.10"
    ) else if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
        set "PYTHON=%LocalAppData%\Programs\Python\Python310\python.exe"
    ) else (
        echo [ERROR] Python 3.10 not found.
        echo         Install Python 3.10 or create .venv in this folder.
        pause
        exit /b 1
    )
)

echo.
echo  Chessy - desktop overlay
echo  ========================
echo.

echo Checking dependencies...
%PYTHON% -c "import PIL, torch, cv2" 2>nul
if errorlevel 1 (
    echo [ERROR] Missing PIL, torch, or opencv.
    echo         Run: py -3.10 -m pip install -e .
    pause
    exit /b 1
)

%PYTHON% -c "import PySide6" 2>nul
if errorlevel 1 (
    echo [ERROR] Missing PySide6 (Qt UI).
    echo         Run: py -3.10 -m pip install "PySide6>=6.6"
    echo         or:  py -3.10 -m pip install -e .
    pause
    exit /b 1
)

echo Starting Chessy overlay...
%PYTHON% -m src.app.ui_qt
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Chessy exited with code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
