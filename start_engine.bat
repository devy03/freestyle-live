@echo off
REM Starts the ACE-Step music engine (API mode) for the freestyle project.
REM ACE-Step's own planner model is turned off to save VRAM - Ollama writes the lyrics instead.
title ACE-Step Engine
cd /d "%~dp0ACE-Step-1.5"
set ACESTEP_INIT_LLM=false
set ACESTEP_CONFIG_PATH=acestep-v15-turbo
set ACESTEP_OFFLOAD_TO_CPU=true
echo Starting ACE-Step engine on http://127.0.0.1:8001 ...
echo Leave this window open while freestyling.
"%~dp0ACE-Step-1.5\python_embeded\python.exe" "%~dp0ACE-Step-1.5\acestep\api_server.py" --host 127.0.0.1 --port 8001
pause
