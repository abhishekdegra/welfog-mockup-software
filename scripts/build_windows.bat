@echo off
REM Build a portable Windows folder under dist\PhoneCoverMockupStudio
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create a venv and install requirements first.
  exit /b 1
)
.venv\Scripts\python.exe -m pip install -q pyinstaller
.venv\Scripts\pyinstaller.exe --noconfirm build.spec
echo.
echo Portable build: dist\PhoneCoverMockupStudio\
exit /b %ERRORLEVEL%
