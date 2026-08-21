param(
    [int]$RetentionDays = 14,
    [int]$AudioChunkSeconds = 15,
    [switch]$Restart
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot 'listener\.env'
$dataDir = Join-Path $env:USERPROFILE '.screenpipe'

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Set-EnvValue([string]$Path, [string]$Name, [string]$Value) {
    if (-not $Value) {
        return
    }
    $content = if (Test-Path -LiteralPath $Path) {
        Get-Content -LiteralPath $Path -Raw
    }
    else {
        ''
    }
    $line = "$Name=$Value"
    $pattern = "(?m)^$([regex]::Escape($Name))=.*$"
    if ($content -match $pattern) {
        $content = [regex]::Replace($content, $pattern, $line)
    }
    else {
        if ($content -and -not $content.EndsWith("`n")) {
            $content += "`r`n"
        }
        $content += "$line`r`n"
    }
    Set-Content -LiteralPath $Path -Value $content -Encoding utf8
}

$screenpipeCommand = Get-Command screenpipe -ErrorAction SilentlyContinue
if (-not $screenpipeCommand) {
    throw "Nie znaleziono Screenpipe w PATH. Zainstaluj pakiet screenpipe CLI."
}

$recorders = @(
    Get-CimInstance Win32_Process -Filter "Name='screenpipe.exe'" |
        Where-Object { $_.CommandLine -match '\brecord\b' }
)
if ($recorders.Count -gt 0 -and -not $Restart) {
    if (Test-Port 3030) {
        Write-Output 'Screenpipe juz dziala.'
        exit 0
    }
    Write-Warning 'Screenpipe dziala, ale port 3030 nie nasluchuje. Restartuje.'
    $Restart = $true
}
if ($Restart) {
    foreach ($recorder in $recorders) {
        Stop-Process -Id $recorder.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline -and (Test-Port 3030)) {
        Start-Sleep -Milliseconds 250
    }
}

$env:SCREENPIPE_KEEP_NORMAL_PRIORITY = '1'

$token = (& screenpipe auth token 2>$null | Out-String).Trim()
if ($token) {
    Set-EnvValue -Path $envPath -Name 'SCREENPIPE_API_TOKEN' -Value $token
}

$deviceLines = @(& screenpipe audio list 2>$null)
$devices = @(
    $deviceLines |
        Where-Object { $_ -match '^\s{2,}\S' } |
        ForEach-Object { $_.Trim() }
)
$preferredInput = $null
$audioProbe = Join-Path $projectRoot 'listener\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $audioProbe) {
    $env:PYTHONIOENCODING = 'utf-8'
    $probeCode = (
        'import soundcard as sc; m=sc.default_microphone(); ' +
        'print(m.name if m else None)'
    )
    $defaultInput = (
        & $audioProbe -c $probeCode 2>$null | Out-String
    ).Trim()
    if ($defaultInput) {
        $preferredInput = $devices |
            Where-Object { $_ -eq "$defaultInput (input)" } |
            Select-Object -First 1
    }
}
if (-not $preferredInput) {
    $preferredInput = $devices |
        Where-Object { $_ -match '\(input\)$' -and $_ -notmatch 'Steam Streaming' } |
        Select-Object -First 1
}
$physicalOutputs = @(
    $devices |
        Where-Object { $_ -match '\(output\)$' -and $_ -notmatch 'Steam Streaming' }
)
$audioDevices = @($preferredInput) + $physicalOutputs |
    Where-Object { $_ } |
    Select-Object -Unique
if ($audioDevices.Count -eq 0) {
    throw 'Nie znaleziono fizycznego wejscia ani wyjscia audio dla Screenpipe.'
}

$arguments = @(
    'record',
    '--port', '3030',
    '--data-dir', $dataDir,
    '--audio-chunk-duration', [string][Math]::Max(10, $AudioChunkSeconds),
    '--audio-transcription-engine', 'disabled',
    '--use-system-default-audio=false',
    '--language', 'polish',
    '--app-context', 'both',
    '--disable-telemetry',
    '--api-auth',
    '--use-pii-removal=true',
    '--retention-days', [string][Math]::Max(1, $RetentionDays),
    '--retention-mode', 'all'
)
foreach ($device in $audioDevices) {
    $arguments += @('--audio-device', $device)
}

Write-Output (
    (
        "Uruchamiam Screenpipe z zachowaniem ustawien prywatnosci uzytkownika, " +
        "redakcja PII, mikrofonem i {0} wyjsciami audio; retencja {1} dni. " +
        "Transkrypcja globalna jest wylaczona. Skonfiguruj ignorowane okna " +
        "w Screenpipe przed przechwytywaniem danych wrazliwych."
    ) -f $physicalOutputs.Count, $RetentionDays
)

& screenpipe @arguments
