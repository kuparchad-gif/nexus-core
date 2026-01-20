# Utilities\viren\schedule_comfort_oscillate.ps1
$proj   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py     = Join-Path $proj ".venv\Scripts\python.exe"
$script = Join-Path $proj "Systems\viren\services\comfort_oscillate.py"

if (-not (Test-Path $py)) {
  Write-Host "Bootstrapping venv..."
  py -3 -m venv (Join-Path $proj ".venv")
  & (Join-Path $proj ".venv\Scripts\python.exe") -m pip install --upgrade pip
  & (Join-Path $proj ".venv\Scripts\python.exe") -m pip install pyyaml
}

$action = New-ScheduledTaskAction -Execute $py -Argument "`"$script`" --tick" -WorkingDirectory $proj
$trigs  = @(
  New-ScheduledTaskTrigger -Daily -At 10:30AM
  New-ScheduledTaskTrigger -Daily -At 10:30PM
)
Register-ScheduledTask -TaskName "Nexus Comfort Oscillate" -Action $action -Trigger $trigs -RunLevel Highest -Force
Write-Host "Registered 'Nexus Comfort Oscillate' with 2 daily triggers."
