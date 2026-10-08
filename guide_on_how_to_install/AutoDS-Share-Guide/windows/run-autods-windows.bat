@echo off
rem Run AutoDS from source in its own window.
rem Put this file inside the AutoDS project folder (next to desktop.py) and double-click it.
rem   run-autods-windows.bat reinstall    reinstall the Python packages
setlocal
cd /d "%~dp0"

if not exist desktop.py (
  echo desktop.py not found. Put this file inside the AutoDS project folder, next to desktop.py.
  pause & exit /b 1
)

if /i "%~1"=="reinstall" del ".venv\.autods_deps_ok" 2>nul

if not exist ".venv\Scripts\python.exe" (
  set "PYCMD="
  for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined PYCMD (
      py -%%V --version >nul 2>nul && set "PYCMD=py -%%V"
    )
  )
  if not defined PYCMD (
    python --version >nul 2>nul && set "PYCMD=python"
  )
  if not defined PYCMD (
    echo AutoDS needs Python 3.10 to 3.13 ^(3.12 is best^).
    echo Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH".
    pause & exit /b 1
  )
  echo Creating environment with %PYCMD% ...
  %PYCMD% -m venv .venv
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)

if not exist ".venv\.autods_deps_ok" (
  echo Installing packages ^(first run takes several minutes^) ...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
  if errorlevel 1 ( echo Package install failed. See TROUBLESHOOTING.md. & pause & exit /b 1 )
  rem requirements-desktop.txt also lists build-only tools; install just the window library
  ".venv\Scripts\python.exe" -m pip install "pywebview>=5.3"
  if errorlevel 1 ( echo Window library install failed. & pause & exit /b 1 )
  echo ok> ".venv\.autods_deps_ok"
)

echo Starting AutoDS ... ^(close this window to quit^)
".venv\Scripts\python.exe" desktop.py
if errorlevel 1 pause
