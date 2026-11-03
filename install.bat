@echo off
title Freestyle Live - setup
cd /d "%~dp0"
set "PY=%~dp0ACE-Step-1.5\python_embeded\python.exe"
if not exist "%PY%" (
  echo.
  echo  ACE-Step was not found.
  echo  1. Download the ACE-Step 1.5 Windows portable package:
  echo       https://files.acemusic.ai/acemusic/win/ACE-Step-1.5.7z
  echo  2. Extract it into this folder, so this file exists:
  echo       %PY%
  echo  3. Run install.bat again.
  echo.
  pause
  exit /b 1
)
echo Installing the vocal separator (Demucs) into ACE-Step's Python...
"%PY%" -m pip install --no-deps demucs==4.1.0 julius==0.2.8 lameenc==1.9.0 sphn==0.2.1
if errorlevel 1 (
  echo Installing Demucs failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo.
echo Downloading the lyric model (about 3 GB)...
where ollama >nul 2>&1
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull gemma3:4b
  ) else (
    echo Ollama is not installed. Get it from https://ollama.com, then run:  ollama pull gemma3:4b
  )
) else (
  ollama pull gemma3:4b
)
echo.
echo Setup is done. Double-click start_everything.bat to start the show.
echo The first start downloads the ACE-Step music models (about 10 GB), so give it time.
pause
