<#
.SYNOPSIS
    Registers the two Windows Task Scheduler tasks that run code/daily.py (Phase 11).

.DESCRIPTION
    Run it yourself from PowerShell (no admin needed for the default):

        powershell -ExecutionPolicy Bypass -File scripts\install_tasks.ps1

    Re-running it replaces both tasks (e.g. with other times). Remove them with scripts\uninstall_tasks.ps1.

    Tasks, in the Task Scheduler folder "\NBA prediction model\":
      Daily update    every day at -DailyAt (default 09:00, local time): code\daily.py
                      (apply finished games, morning lines snapshot, rebuild the site)
      Pregame lines   every -PregameEveryMinutes (15) minutes for -PregameHours (14) hours from -PregameFrom
                      (10:00 local): code\daily.py --pregame. Each run exits at once until the snapshot is due
                      (1 hour before the day's first tip-off, Eastern; 6 PM ET if unknown), then takes ONE
                      snapshot for the day.

    Both use the repo's venv\Scripts\pythonw.exe (no console window; everything goes to logs\daily_YYYY-MM-DD.log),
    run from the repo folder, and:
      * run whether or not VS Code is open (Task Scheduler starts them, not the editor)
      * catch up after sleep / shutdown: "run as soon as possible after a scheduled start is missed"
      * run on battery and keep running if the PC unplugs
      * never overlap: a new start is ignored while one is running (and daily.py has its own lock)
      * get ODDS_API_KEY from your user environment (set once with
        [Environment]::SetEnvironmentVariable("ODDS_API_KEY", "...", "User"); daily.py also reads it from
        the registry if the task's environment predates it). The key is never written into the task.

    By default the tasks run only while you are logged on (the usual case on a home PC; no password stored).
    -WhenLoggedOff registers them to run whether you are logged on or not (S4U, no password stored); that
    needs an elevated (Run as administrator) PowerShell.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_tasks.ps1 -DailyAt 08:30 -PregameFrom 11:00
#>
[CmdletBinding()]
param(
    [string]$DailyAt = "09:00",
    [string]$PregameFrom = "10:00",
    [ValidateRange(1, 24)][int]$PregameHours = 14,
    [ValidateRange(5, 120)][int]$PregameEveryMinutes = 15,
    [switch]$WhenLoggedOff
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo "venv\Scripts\pythonw.exe"
$Script = Join-Path $Repo "code\daily.py"
$TaskPath = "\NBA prediction model\"
$DailyName = "Daily update"
$PregameName = "Pregame lines"

if (-not (Test-Path $Python)) { throw "No venv python at $Python; create the venv first (python -m venv venv; pip install -r requirements.txt)" }
if (-not (Test-Path $Script)) { throw "No $Script; run this script from the repo's scripts folder" }

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$logon = if ($WhenLoggedOff) { "S4U" } else { "Interactive" }
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType $logon -RunLevel Limited

function New-Settings([int]$Minutes) {
    New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $Minutes)
}

# daily update
$dailyAction = New-ScheduledTaskAction -Execute $Python -Argument "`"$Script`"" -WorkingDirectory $Repo
$dailyTrigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
Register-ScheduledTask -TaskPath $TaskPath -TaskName $DailyName -Action $dailyAction -Trigger $dailyTrigger `
    -Settings (New-Settings 60) -Principal $principal -Force `
    -Description "NBA prediction model: apply finished games, morning betting lines, rebuild the site (code\daily.py)." | Out-Null

# pre-game lines: a daily trigger repeated through the day (the usual PowerShell 5.1 way to set a repetition)
$pregameAction = New-ScheduledTaskAction -Execute $Python -Argument "`"$Script`" --pregame" -WorkingDirectory $Repo
$pregameTrigger = New-ScheduledTaskTrigger -Daily -At $PregameFrom
$repeat = New-ScheduledTaskTrigger -Once -At $PregameFrom -RepetitionInterval (New-TimeSpan -Minutes $PregameEveryMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours $PregameHours)
$pregameTrigger.Repetition = $repeat.Repetition
Register-ScheduledTask -TaskPath $TaskPath -TaskName $PregameName -Action $pregameAction -Trigger $pregameTrigger `
    -Settings (New-Settings 30) -Principal $principal -Force `
    -Description "NBA prediction model: closing-line snapshot about 1 hour before the first tip-off (code\daily.py --pregame)." | Out-Null

Write-Host "Registered in Task Scheduler, folder $TaskPath (runs as $user, logon $logon):"
Write-Host "  $DailyName    daily at $DailyAt"
Write-Host "  $PregameName  every $PregameEveryMinutes min for $PregameHours h from $PregameFrom (snapshot only when due)"
Write-Host "  python: $Python"
Write-Host "  logs:   $(Join-Path $Repo 'logs')\daily_YYYY-MM-DD.log"
if (-not [Environment]::GetEnvironmentVariable("ODDS_API_KEY", "User")) {
    Write-Warning "ODDS_API_KEY is not in your user environment: the runs will skip betting lines. Set it with [Environment]::SetEnvironmentVariable('ODDS_API_KEY', '...', 'User')"
}
Write-Host ""
Write-Host "Try it now:  Start-ScheduledTask -TaskPath '$TaskPath' -TaskName '$DailyName'"
Write-Host "Check it:    Get-ScheduledTaskInfo -TaskPath '$TaskPath' -TaskName '$DailyName'   (LastTaskResult = daily.py's exit code)"
