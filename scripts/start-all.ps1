[CmdletBinding()]
param(
    [ValidateRange(30, 1800)]
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Port 1234)) {
    Write-Warning 'LM Studio API nie nasluchuje na porcie 1234. Uruchom Local Server w LM Studio.'
}

$syncUiVision = Join-Path $PSScriptRoot 'sync-uivision.ps1'
if (Test-Path $syncUiVision -PathType Leaf) {
    try {
        & $syncUiVision | Write-Output
    } catch {
        Write-Warning "Nie udalo sie zsynchronizowac UI.Vision: $($_.Exception.Message)"
    }
}

if (-not (Test-Port 6333)) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSScriptRoot\start-qdrant.ps1`""
    ) -WindowStyle Hidden
}

if (-not (Test-Port 3030)) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSScriptRoot\start-screenpipe.ps1`""
    ) -WindowStyle Hidden
}

# n8n jest wylaczone (N8N_ENABLED=false)
# if (-not (Test-Port 5678)) {
#     Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$PSScriptRoot\start-n8n.bat`""
# }

if (-not (Test-Port 8765)) {
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$projectRoot\listener\start-listener.bat`""
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (
        (Test-Port 3030) -and
        (Test-Port 6333) -and
        (Test-Port 8765)
    ) {
        Start-Process 'http://127.0.0.1:8765'
        Write-Output 'VoiceLoop uruchomiony.'
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

throw "VoiceLoop nie uruchomil wszystkich uslug w $TimeoutSeconds sekund. Sprawdz porty i logi."
