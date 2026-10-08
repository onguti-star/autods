# Creates an "AutoDS" icon on your Desktop that starts the app without a console window.
# Put this file in the AutoDS project folder, run run-autods-windows.bat once first,
# then right-click this file and choose "Run with PowerShell"
# (or: powershell -ExecutionPolicy Bypass -File make-desktop-shortcut-windows.ps1).
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw  = Join-Path $here ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) {
    Write-Host "Run run-autods-windows.bat once first, so the environment exists."
    exit 1
}
$desktop = [Environment]::GetFolderPath("Desktop")
$ws  = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desktop "AutoDS.lnk"))
$lnk.TargetPath       = $pyw
$lnk.Arguments        = '"' + (Join-Path $here "desktop.py") + '"'
$lnk.WorkingDirectory = $here
$lnk.IconLocation     = (Join-Path $here "assets\autods.ico")
$lnk.Description      = "AutoDS"
$lnk.Save()
Write-Host "Done. Look for AutoDS on your Desktop."
