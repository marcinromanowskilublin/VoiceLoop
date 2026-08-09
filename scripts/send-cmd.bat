@echo off
rem VoiceLoop: wysyla dozwolona komende do lokalnego rdzenia.
if "%~1"=="" (
  echo Uzycie: send-cmd.bat ^<id_komendy^>
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0send-command.ps1" -CommandId "%~1"
exit /b %ERRORLEVEL%
