# VoiceLoop: bezpieczny runner UI.Vision Command Line API.
param(
    [Parameter(Mandatory = $true)][string]$Macro,
    [string]$Var1 = '',
    [string]$Var2 = '',
    [string]$Var3 = '',
    [ValidateRange(5, 600)][int]$TimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'

if (
    $Macro.Length -gt 160 -or
    $Macro.Contains('..') -or
    $Macro -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$' -or
    [System.IO.Path]::GetFileName($Macro) -ne $Macro
) {
    Write-Error 'Nieprawidlowa nazwa makra.'
    exit 2
}

$chrome = if ($env:CHROME_PATH) { $env:CHROME_PATH } else {
    'C:\Program Files\Google\Chrome\Application\chrome.exe'
}
$uivisionHome = if ($env:UIVISION_HOME) { $env:UIVISION_HOME } else {
    Join-Path $env:USERPROFILE 'Desktop\uivision'
}
$page = Join-Path $uivisionHome 'ui.vision.html'
$macrosRoot = [System.IO.Path]::GetFullPath((Join-Path $uivisionHome 'macros'))
$macroPath = [System.IO.Path]::GetFullPath((Join-Path $macrosRoot $Macro))
$macrosPrefix = $macrosRoot.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
if (-not $macroPath.StartsWith($macrosPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error 'Makro musi znajdowac sie bezposrednio w katalogu runtime macros.'
    exit 2
}

if (-not (Test-Path $chrome -PathType Leaf)) {
    Write-Error "Brak Chrome: $chrome"
    exit 3
}
if (-not (Test-Path $page)) {
    Write-Error "Brak pliku $page. W UI.Vision: Settings -> API -> 'Create autorun HTML page', zapisz plik do $uivisionHome"
    exit 4
}
if (-not (Test-Path $macroPath -PathType Leaf)) {
    Write-Error "Brak makra runtime: $macroPath. Uruchom scripts\sync-uivision.ps1."
    exit 5
}

$pagePath = $page -replace '\\', '/'
$projectRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$runId = [guid]::NewGuid().ToString('N')
$logFile = Join-Path $logsDir "uivision_$runId.txt"
$logPath = $logFile -replace '\\', '/'

$encodedMacro = [uri]::EscapeDataString(($Macro -replace '\\', '/'))
$url = "file:///${pagePath}?storage=xfile&macro=$encodedMacro&direct=1&closeRPA=1&closeBrowser=1&savelog=$logPath&continueInLastUsedTab=0"
if ($Var1) { $url += '&cmd_var1=' + [uri]::EscapeDataString($Var1) }
if ($Var2) { $url += '&cmd_var2=' + [uri]::EscapeDataString($Var2) }
if ($Var3) { $url += '&cmd_var3=' + [uri]::EscapeDataString($Var3) }

Start-Process -FilePath $chrome -ArgumentList '--new-window', "`"$url`"" | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-Path $logFile -PathType Leaf) {
        $content = Get-Content -Raw -Encoding UTF8 $logFile
        $firstLine = ($content -split "`r?`n", 2)[0]
        if ($firstLine -match '(?i)(error|failed|status\s*=\s*false)') {
            Write-Error "UI.Vision: $firstLine (log: $logFile)"
            exit 10
        }
        Write-Output "UI.Vision OK: $Macro (log: $logFile)"
        exit 0
    }
    Start-Sleep -Milliseconds 250
}

Write-Error "Timeout UI.Vision po $TimeoutSeconds s: $Macro"
exit 124
