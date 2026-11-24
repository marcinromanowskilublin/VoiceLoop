# Watchdog for local calibration server. Run under current Windows user session.
param([switch]$Once)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root "listener\.venv\Scripts\python.exe"
$Prefix = @()
if (-not (Test-Path $Python)) {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $Python = $launcher.Source
        $Prefix = @("-3.11")
    } else {
        $Python = (Get-Command python.exe -ErrorAction Stop).Source
    }
}
$ServerScript = Join-Path $Root "scripts\calibration-phrases\server.py"
$Mutex = [System.Threading.Mutex]::new($false, "Local\VoiceLoopMinimalizuj8792")
$HasLock = $false
try {
    $HasLock = $Mutex.WaitOne(0)
    if (-not $HasLock) { Write-Output "VoiceLoop Minimalizuj watchdog is already running."; exit 0 }
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
    do {
        try {
            $Args = @($Prefix) + @('"' + $ServerScript + '"')
            $Proc = Start-Process -FilePath $Python -ArgumentList $Args -PassThru -WindowStyle Hidden -WorkingDirectory $Root -RedirectStandardOutput (Join-Path $Root "logs\minimalizuj.stdout.log") -RedirectStandardError (Join-Path $Root "logs\minimalizuj.stderr.log")
            $Misses = 0
            while (-not $Proc.HasExited) {
                Start-Sleep -Seconds 10
                if ($Proc.HasExited) { break }
                try {
                    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8792/api/health" -TimeoutSec 3
                    if ($Health.ok -and $Health.tool -eq "calibration-phrases") {
                        $Misses = 0
                    } else {
                        $Misses++
                    }
                } catch {
                    $Misses++
                }
                if ($Misses -ge 3) {
                    Write-Warning "Local calibration health failed 3 times; restarting owned process."
                    Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
                    break
                }
            }
            $Proc.WaitForExit()
            Write-Output "Minimalizuj exited: $($Proc.ExitCode)"
        } catch {
            Write-Warning $_
        }
        if (-not $Once) { Start-Sleep -Seconds 5 }
    } while (-not $Once)
} finally {
    if ($HasLock) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
