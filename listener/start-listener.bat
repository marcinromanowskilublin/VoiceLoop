@echo off
title VoiceLoop - Local Assistant Core
cd /d "%~dp0"
if not exist .venv (
  echo [setup] Tworze srodowisko Python 3.11...
  py -3.11 -m venv .venv 2>nul || py -m venv .venv
  call ".venv\Scripts\python" -m pip install -q --upgrade pip
)
if exist requirements.lock (
  call ".venv\Scripts\pip" install -q -r requirements.lock
) else (
  call ".venv\Scripts\pip" install -q -r requirements.in
)
echo VoiceLoop: http://127.0.0.1:8765
call ".venv\Scripts\python" -m uvicorn voiceloop.app:app --host 127.0.0.1 --port 8765
pause
