param(
    [int]$RetentionDays = 14,
    [int]$AudioChunkSeconds = 15,
    [int]$IdleCaptureIntervalMs = 5000,
    [int]$VisualCheckIntervalMs = 1000,
    [int]$MinCaptureIntervalMs = 1500,
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

function Set-UnfilteredCaptureSettings([string]$StorePath) {
    if (-not (Test-Path -LiteralPath $StorePath)) {
        return
    }
    $store = Get-Content -LiteralPath $StorePath -Raw | ConvertFrom-Json
    if (-not $store.settings) {
        return
    }
    $values = [ordered]@{
        appContext = 'both'
        asyncImagePiiRedaction = $false
        asyncPiiRedaction = $false
        audioCaptureMode = 'always'
        captureOnClipboard = $true
        captureOnKeystroke = $true
        captureScroll = $true
        disableClickCapture = $false
        disableClipboardCapture = $false
        disableKeyboardCapture = $false
        enhancedIncognitoDetection = $false
        idleCaptureIntervalMs = $IdleCaptureIntervalMs
        ignoreIncognitoWindows = $false
        minCaptureIntervalMs = $MinCaptureIntervalMs
        pauseOnDrmContent = $false
        piiRedactionPseudonyms = $false
        redactAgentSessionSecrets = $false
        scheduleEnabled = $false
        usePiiRemoval = $false
        visualCheckIntervalMs = $VisualCheckIntervalMs
    }
    foreach ($entry in $values.GetEnumerator()) {
        $store.settings |
            Add-Member -MemberType NoteProperty -Name $entry.Key -Value $entry.Value -Force
    }
    foreach (
        $name in @('ignoredMeetingApps', 'ignoredUrls', 'ignoredWindows', 'includedWindows')
    ) {
        $store.settings |
            Add-Member -MemberType NoteProperty -Name $name -Value @() -Force
    }
    $json = $store | ConvertTo-Json -Depth 100 -Compress
    [System.IO.File]::WriteAllText(
        $StorePath,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )
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

Set-UnfilteredCaptureSettings -StorePath (Join-Path $dataDir 'store.bin')
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
    '--use-all-monitors',
    '--language', 'polish',
    '--app-context', 'both',
    '--ignore-incognito-windows=false',
    '--disable-telemetry',
    '--api-auth',
    '--use-pii-removal=false',
    '--retention-days', [string][Math]::Max(1, $RetentionDays),
    '--retention-mode', 'all',
    '--capture-on-keystroke', 'true',
    '--capture-on-clipboard', 'true',
    '--capture-scroll', 'true',
    '--idle-capture-interval-ms', [string][Math]::Max(1000, $IdleCaptureIntervalMs),
    '--visual-check-interval-ms', [string][Math]::Max(250, $VisualCheckIntervalMs),
    '--min-capture-interval-ms', [string][Math]::Max(500, $MinCaptureIntervalMs)
)
foreach ($device in $audioDevices) {
    $arguments += @('--audio-device', $device)
}

Write-Output (
    (
        "Uruchamiam Screenpipe: 2 monitory, mikrofon i {0} wyjsc audio, " +
        "pelne zdarzenia wejscia, OCR co najmniej co {1}s, retencja {2} dni. " +
        "Transkrypcja globalna wylaczona."
    ) -f $physicalOutputs.Count, [Math]::Round($IdleCaptureIntervalMs / 1000, 1), $RetentionDays
)

& screenpipe @arguments
