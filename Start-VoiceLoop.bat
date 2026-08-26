@echo off
title VoiceLoop - uruchamianie calego stosu
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-all.ps1"
if errorlevel 1 (
  echo.
  echo VoiceLoop nie wystartowal poprawnie. Szczegoly sa powyzej.
  pause
  exit /b 1
)
exit /b 0
