' VoiceLoop: "take a note" -> lokalny rdzen -> UI.Vision
Set fso = CreateObject("Scripting.FileSystemObject")
scriptsDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptsDir & "\send-command.ps1"" -CommandId note", 0, False
