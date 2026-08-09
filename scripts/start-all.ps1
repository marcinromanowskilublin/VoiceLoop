$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Port 1234)) {
    Write-Warning 'LM Studio API nie nasluchuje na porcie 1234. Uruchom Local Server w LM Studio.'
}

if (-not (Test-Port 5678)) {
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$PSScriptRoot\start-n8n.bat`""
}

if (-not (Test-Port 8765)) {
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$projectRoot\listener\start-listener.bat`""
}

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    if ((Test-Port 5678) -and (Test-Port 8765)) {
        Start-Process 'http://127.0.0.1:8765'
        Write-Output 'VoiceLoop uruchomiony.'
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

throw 'VoiceLoop nie uruchomil wszystkich uslug w 30 sekund.'
