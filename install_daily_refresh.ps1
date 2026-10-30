param(
    [string]$TaskName = "MedicineDemandDailyRefresh",
    [string]$RunAt = "02:00"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $Project "config.json"
if (-not (Test-Path -LiteralPath $ConfigPath)) { throw "Open the dashboard and save a public HTTPS CSV source first." }
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if (-not $Config.data_url.StartsWith("https://")) { throw "The saved CSV source must start with https://." }
$Python = (Get-Command python -ErrorAction Stop).Source
$Runner = Join-Path $Project "refresh_and_train.py"
$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Runner`"" -WorkingDirectory $Project
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Fetch configured pharmacy CSV feed and retrain demand models" -Force | Out-Null
Write-Host "Installed daily task '$TaskName' at $RunAt. It will fetch the saved CSV source and retrain only when data changes."
