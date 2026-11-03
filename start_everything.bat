@echo off
title Freestyle Live - starter
cd /d "%~dp0"
echo Starting the music engine in its own window...
start "ACE-Step Engine" cmd /c "%~dp0start_engine.bat"
echo Waiting for the engine to be ready (this can take a minute)...
:wait
timeout /t 3 /nobreak >nul
curl -s http://127.0.0.1:8001/health >nul 2>&1
if errorlevel 1 goto wait
echo Engine is ready. Starting the website...
start "Freestyle Live" cmd /c "%~dp0run_live.bat"
exit
