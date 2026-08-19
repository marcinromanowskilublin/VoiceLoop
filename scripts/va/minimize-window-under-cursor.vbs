' VoiceLoop v2: minimize the application window under the mouse cursor
Set fso = CreateObject("Scripting.FileSystemObject")
scriptsDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & scriptsDir & "\send-command.ps1"" -CommandId minimize_window_under_cursor", 0, False
