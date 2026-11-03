@echo off
title Freestyle Live - public link
cd /d "%~dp0"
if not exist "%~dp0tools\cloudflared.exe" (
  echo Downloading Cloudflare's tunnel tool (one time, about 55 MB^)...
  if not exist "%~dp0tools" mkdir "%~dp0tools"
  curl -L -o "%~dp0tools\cloudflared.exe" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)
echo.
echo  Making a public link to your Freestyle Live website...
echo  Look below for a line ending in .trycloudflare.com  - that is the link to share.
echo  The link works while this window, the engine, and run_live.bat are all open.
echo  Close this window to take the site offline. You get a new link each time.
echo.
"%~dp0tools\cloudflared.exe" tunnel --url http://127.0.0.1:5050 --no-autoupdate
pause
