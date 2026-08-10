' VoiceLoop v2: capture one fact, then require confirmation
Set fso = CreateObject("Scripting.FileSystemObject")
scriptsDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptsDir & "\send-command.ps1"" -Operation listen-once -Mode remember", 0, False
