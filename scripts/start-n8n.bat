@echo off
title n8n - VoiceLoop
set N8N_LISTEN_ADDRESS=127.0.0.1
set N8N_HOST=127.0.0.1
set NODES_EXCLUDE=["n8n-nodes-base.executeCommand"]
set N8N_ENABLE_EXECUTE_COMMAND=false
echo ============================================
echo  n8n startuje lokalnie: http://127.0.0.1:5678
echo  Router: POST /webhook/voice-command-v1
echo  Akcje wykonuje bezpieczny rdzen VoiceLoop, nie shell n8n.
echo ============================================
call "%APPDATA%\npm\n8n.cmd"
pause
