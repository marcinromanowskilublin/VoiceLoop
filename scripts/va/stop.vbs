' VoiceLoop: "stop now" -> bezposredni STOP rdzenia
Set fso = CreateObject("Scripting.FileSystemObject")
scriptsDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptsDir & "\send-command.ps1"" -CommandId stop", 0, False
