# Utilities\viren\run_comfort_oscillate.ps1
param(
  [switch]$Tick,
  [string]$DecisionId,
  [ValidateSet('ARCHIVE_NOW','RETAIN','FEEL_AGAIN','POSTPONE')]
  [string]$Choice
)
$ErrorActionPreference = "Stop"
$proj   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py     = Join-Path $proj ".venv\Scripts\python.exe"
$script = Join-Path $proj "Systems\viren\services\comfort_oscillate.py"

if (-not (Test-Path $py)) {
  Write-Host "Bootstrapping venv..."
  py -3 -m venv (Join-Path $proj ".venv")
  & (Join-Path $proj ".venv\Scripts\python.exe") -m pip install --upgrade pip
  & (Join-Path $proj ".venn\Scripts\python.exe") -m pip install pyyaml
}

if ($Tick) {
  & $py $script --tick
  exit $LASTEXITCODE
}
if ($DecisionId -and $Choice) {
  & $py $script --decision $DecisionId $Choice
  exit $LASTEXITCODE
}
Write-Host "Usage:"
Write-Host "  .\run_comfort_oscillate.ps1 -Tick"
Write-Host "  .\run_comfort_oscillate.ps1 -DecisionId <id> -Choice <ARCHIVE_NOW|RETAIN|FEEL_AGAIN|POSTPONE>"
