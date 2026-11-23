# Installs an interactive per-user scheduled task, without administrator privileges.
$ErrorActionPreference = "Stop"
$Script = (Resolve-Path (Join-Path $PSScriptRoot "run-minimalizuj.ps1")).Path
$Exe = (Get-Command powershell.exe -ErrorAction Stop).Source
$Name = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $Exe -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $Script + '"')
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $Name
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $Name -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "VoiceLoop Minimalizuj" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Start-ScheduledTask -TaskName "VoiceLoop Minimalizuj"
Write-Output "Installed. Check http://127.0.0.1:8792/api/health"
