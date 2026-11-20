<#
.SYNOPSIS
    Removes the two Task Scheduler tasks registered by scripts\install_tasks.ps1 (Phase 11).

.DESCRIPTION
        powershell -ExecutionPolicy Bypass -File scripts\uninstall_tasks.ps1

    Deletes "\NBA prediction model\Daily update" and "\NBA prediction model\Pregame lines" (whichever exist)
    and then the empty task folder. Nothing in the repo is touched (logs\ stays).
#>
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$TaskPath = "\NBA prediction model\"
foreach ($name in @("Daily update", "Pregame lines")) {
    $task = Get-ScheduledTask -TaskPath $TaskPath -TaskName $name -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $name -Confirm:$false
        Write-Host "Removed $TaskPath$name"
    } else {
        Write-Host "Not registered: $TaskPath$name"
    }
}

# the folder itself, if it is now empty
try {
    $service = New-Object -ComObject Schedule.Service
    $service.Connect()
    $folder = $service.GetFolder($TaskPath.TrimEnd("\"))
    if ($folder.GetTasks(1).Count -eq 0 -and $folder.GetFolders(0).Count -eq 0) {
        $service.GetFolder("\").DeleteFolder($TaskPath.Trim("\"), 0)
        Write-Host "Removed the empty folder $TaskPath"
    }
} catch {
    # no such folder (never installed) or not empty: nothing to do
}
