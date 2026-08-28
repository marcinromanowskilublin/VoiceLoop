[CmdletBinding()]
param(
    [ValidateRange(30, 1800)]
    [int]$TimeoutSeconds = 300,
    [switch]$NoPanel,
    [switch]$NoVoiceAttack,
    [switch]$FullScreenpipeCapture,
    [switch]$RunSmokeTest
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$listenerRoot = Join-Path $projectRoot 'listener'

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Test-Http([string]$Uri, [int]$Timeout = 3) {
    try {
        Invoke-RestMethod -Uri $Uri -TimeoutSec $Timeout | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Get-LocalEnvValue([string]$Name) {
    $envPath = Join-Path $listenerRoot '.env'
    if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
        return $null
    }
    $match = Get-Content -LiteralPath $envPath -Encoding UTF8 |
        Where-Object { $_ -match "^$([regex]::Escape($Name))=" } |
        Select-Object -Last 1
    if (-not $match) {
        return $null
    }
    return ($match -split '=', 2)[1].Trim().Trim('"').Trim("'")
}

if (-not (Test-Port 1234)) {
    $lmStudioExe = Join-Path $env:LOCALAPPDATA 'Programs\LM Studio\LM Studio.exe'
    if (
        (Test-Path -LiteralPath $lmStudioExe -PathType Leaf) -and
        -not (Get-Process -Name 'LM Studio' -ErrorAction SilentlyContinue)
    ) {
        Start-Process -FilePath $lmStudioExe
        Write-Output 'Uruchomiono LM Studio; czekam na lokalny serwer API.'
    }
    else {
        Write-Warning (
            'LM Studio API nie nasluchuje na porcie 1234. ' +
            'W LM Studio wlacz Developer -> Local Server.'
        )
    }
}

$syncUiVision = Join-Path $PSScriptRoot 'sync-uivision.ps1'
if (Test-Path $syncUiVision -PathType Leaf) {
    try {
        & $syncUiVision | Write-Output
    } catch {
        Write-Warning "Nie udalo sie zsynchronizowac UI.Vision: $($_.Exception.Message)"
    }
}

$startQdrant = Join-Path $PSScriptRoot 'start-qdrant.ps1'
& $startQdrant | Write-Output

if (-not (Test-Port 3030)) {
    $screenpipeArguments = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSScriptRoot\start-screenpipe.ps1`""
    )
    if (-not $FullScreenpipeCapture) {
        $screenpipeArguments += '-ContextOnly'
    }
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        $screenpipeArguments
    ) -WindowStyle Hidden
}

# n8n jest wylaczone (N8N_ENABLED=false)
# if (-not (Test-Port 5678)) {
#     Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$PSScriptRoot\start-n8n.bat`""
# }

if (-not (Test-Port 8765)) {
    $python = Join-Path $listenerRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $python -PathType Leaf) {
        Start-Process -FilePath $python -ArgumentList @(
            '-m', 'uvicorn',
            'voiceloop.app:app',
            '--host', '127.0.0.1',
            '--port', '8765'
        ) -WorkingDirectory $listenerRoot -WindowStyle Hidden
    }
    else {
        Start-Process -FilePath 'cmd.exe' -ArgumentList @(
            '/c',
            "`"$listenerRoot\start-listener.bat`""
        )
    }
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (
        (Test-Http 'http://127.0.0.1:1234/v1/models') -and
        (Test-Http 'http://127.0.0.1:3030/health') -and
        (Test-Http 'http://127.0.0.1:6333/healthz') -and
        (Test-Http 'http://127.0.0.1:8765/api/v1/health' 15)
    ) {
        break
    }
    Start-Sleep -Milliseconds 500
}

if (
    -not (Test-Http 'http://127.0.0.1:1234/v1/models') -or
    -not (Test-Http 'http://127.0.0.1:3030/health') -or
    -not (Test-Http 'http://127.0.0.1:6333/healthz') -or
    -not (Test-Http 'http://127.0.0.1:8765/api/v1/health' 15)
) {
    throw (
        "VoiceLoop nie uruchomil wszystkich uslug w $TimeoutSeconds sekund. " +
        'Sprawdz LM Studio, Docker, Screenpipe i log listenera.'
    )
}

$voiceAttackState = 'pominiety'
if (-not $NoVoiceAttack) {
    $voiceAttack = Get-Process -Name 'VoiceAttack' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $voiceAttack) {
        $voiceAttackExe = Get-LocalEnvValue 'VOICEATTACK_EXE'
        if (-not $voiceAttackExe) {
            $voiceAttackExe = Join-Path $env:ProgramFiles 'VoiceAttack\VoiceAttack.exe'
        }
        if (-not (Test-Path -LiteralPath $voiceAttackExe -PathType Leaf)) {
            throw "Nie znaleziono VoiceAttack: $voiceAttackExe"
        }
        Start-Process -FilePath $voiceAttackExe
        $voiceAttackDeadline = (Get-Date).AddSeconds(20)
        while (
            (Get-Date) -lt $voiceAttackDeadline -and
            -not (Get-Process -Name 'VoiceAttack' -ErrorAction SilentlyContinue)
        ) {
            Start-Sleep -Milliseconds 250
        }
    }
    $voiceAttackState = if (
        Get-Process -Name 'VoiceAttack' -ErrorAction SilentlyContinue
    ) {
        'uruchomiony'
    }
    else {
        'nie wystartowal'
    }
}

$auditScript = Join-Path $PSScriptRoot 'check_voiceattack_actions.py'
$auditPython = Join-Path $listenerRoot '.venv\Scripts\python.exe'
if (
    -not $NoVoiceAttack -and
    (Test-Path -LiteralPath $auditScript -PathType Leaf) -and
    (Test-Path -LiteralPath $auditPython -PathType Leaf)
) {
    & $auditPython $auditScript | Write-Output
    if ($LASTEXITCODE -ne 0) {
        throw 'Audyt VoiceAttack wykryl niespojnosc profilu lub allowlisty.'
    }
}

if ($RunSmokeTest) {
    & (Join-Path $PSScriptRoot 'test-loop.ps1')
}

if (-not $NoPanel) {
    Start-Process 'http://127.0.0.1:8765'
}

$screenpipeMode = if ($FullScreenpipeCapture) { 'pelny' } else { 'kontekstowy' }
Write-Output ''
Write-Output 'VoiceLoop uruchomiony:'
Write-Output '  LM Studio : OK (127.0.0.1:1234)'
Write-Output "  Screenpipe: OK (tryb $screenpipeMode, 127.0.0.1:3030)"
Write-Output '  Qdrant    : OK (127.0.0.1:6333)'
Write-Output '  Listener  : OK (127.0.0.1:8765)'
Write-Output "  VoiceAttack: $voiceAttackState"
Write-Output '  Panel     : http://127.0.0.1:8765'
