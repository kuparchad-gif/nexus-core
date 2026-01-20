# C:\Projects\Stacks\nexus-metatron\tools\lmstudio_soft_probe.ps1
<#
  Soft probe for LM Studio — *no hypervigilance*
  - Reads/writes var\state\lmstudio_probe_state.json
  - Only probes when cooldown has expired
  - On success: records last_ok and clears cooldown
  - On failure: exponential backoff (15s → up to 10m)

  Usage:
    .\tools\lmstudio_soft_probe.ps1 -RepoRoot "C:\Projects\Stacks\nexus-metatron"
    .\tools\lmstudio_soft_probe.ps1 -Url "http://localhost:1234"
#>
param(
  [string]$RepoRoot = (Get-Location).Path,
  [string]$Url = $env:LMSTUDIO_URL
)

$ErrorActionPreference = "Stop"
if (-not $Url -or $Url.Trim() -eq "") { $Url = "http://localhost:1234" }

$statePath = Join-Path $RepoRoot "var\state\lmstudio_probe_state.json"
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

function Load-State {
  if (Test-Path $statePath) {
    try { return Get-Content $statePath -Raw | ConvertFrom-Json } catch { return @{} }
  }
  return @{}
}
function Save-State($s) {
  $dir = Split-Path -Parent $statePath
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $s | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
function Backoff([hashtable]$s,[long]$now) {
  $fails = [int]($s.fails) + 1
  $windows = 15,45,120,300,600
  $win = if ($fails -le $windows.Length) { $windows[$fails-1] } else { $windows[-1] }
  $s.fails = $fails
  $s.last_fail = $now
  $s.cooldown_until = $now + $win
}
function Mark-Ok([hashtable]$s,[long]$now) {
  $s.fails = 0
  $s.last_ok = $now
  $s.cooldown_until = $now
}

# Only probe if cooldown expired
$state = Load-State
$cooldown = [long]($state.cooldown_until)
if ($now -lt $cooldown) {
  Write-Host "Cooldown active; next probe allowed at $cooldown (epoch)." -ForegroundColor DarkYellow
  exit 0
}

# Normalize URL to /v1/models
$u = $Url.TrimEnd('/')
if ($u -match '/v1$') { $u = "$u/models" } else { $u = "$u/v1/models" }

try {
  $resp = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 2 -Method GET -ErrorAction Stop
  if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
    Mark-Ok -s $state -now $now
    Save-State $state
    Write-Host "LM Studio OK ($($resp.StatusCode)) $u" -ForegroundColor Green
  } else {
    Backoff -s $state -now $now
    Save-State $state
    Write-Host "LM Studio non-OK ($($resp.StatusCode)); backoff engaged." -ForegroundColor Yellow
  }
} catch {
  Backoff -s $state -now $now
  Save-State $state
  Write-Host "LM Studio unreachable; backoff engaged. ($($_.Exception.Message))" -ForegroundColor DarkYellow
}