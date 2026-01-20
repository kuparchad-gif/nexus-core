# metatron_vacuum.ps1 — "vacuum" local files into running Podman pod containers.
# - Copies ONLY existing service folders into containers with the SAME names
# - Excludes junk (node_modules, __pycache__, .venv, .git, etc.)
# - Optional: -InstallDeps runs pip/npm in the container if config files exist
# - Optional: -Restart restarts each container after sync

param(
  [switch]$InstallDeps,
  [switch]$Restart
)

$ErrorActionPreference = "Stop"
$ServicesRoot = 'C:\Projects\Stacks\nexus-metatron\backend\services'
$StageRoot    = Join-Path $env:TEMP 'metatron_stage'

# Map of services you care about (container name == folder name)
$Services = @(
  'gateway',
  'edge-anynode',
  'anynode',
  'planner',
  'llm-proxy',
  'memory',
  'archivist',
  'language-processing',
  'visual_cortex',
  'subconsciousness',
  'consciousness',
  'nexus-core',
  'viren'
)

function Exists([string]$name) {
  try {
    $names = (podman ps -a --format "{{.Names}}") -split "`n"
    return $names -contains $name
  } catch { return $false }
}

function Running([string]$name) {
  try {
    $rows = (podman ps --format "{{.Names}}|{{.Status}}") -split "`n"
    foreach($r in $rows){
      $parts = $r -split '\|',2
      if($parts[0] -eq $name -and $parts[1] -match '^Up\b'){ return $true }
    }
    return $false
  } catch { return $false }
}

function Stage-Copy([string]$src, [string]$dst) {
  if(Test-Path $dst){ Remove-Item -Recurse -Force $dst | Out-Null }
  New-Item -ItemType Directory -Force -Path $dst | Out-Null

  # Robocopy filters
  $xf = @('*.pyc','*.log','*.db','*.sqlite','*.pyo','*.map')
  $xd = @('.git','node_modules','__pycache__','.venv','venv','.idea','.vscode','dist','build')

  $xfArgs = $xf | ForEach-Object { "/XF `"$($_)`"" }
  $xdArgs = $xd | ForEach-Object { "/XD `"$($_)`"" }

  $args = @("$src", "$dst", "/MIR", "/NFL","/NDL","/NJH","/NJS","/NP","/R:1","/W:1") + $xfArgs + $xdArgs
  robocopy @args | Out-Null

  # Robocopy exit codes <=7 are success-ish
  if($LASTEXITCODE -gt 7){
    throw "Robocopy failed ($LASTEXITCODE) for $src"
  }
}

function Sync-One([string]$name){
  $svcDir = Join-Path $ServicesRoot $name
  if(!(Test-Path $svcDir)){ Write-Host " [skip] $name (no folder)" -ForegroundColor DarkGray; return }

  if(!(Exists $name)){ Write-Host " [skip] $name (no container)" -ForegroundColor DarkGray; return }
  if(!(Running $name)){ Write-Host " [skip] $name (container not running)" -ForegroundColor DarkGray; return }

  $stageDir = Join-Path $StageRoot $name
  Write-Host " [+] Staging $name from $svcDir" -ForegroundColor Cyan
  Stage-Copy $svcDir $stageDir

  Write-Host "     -> podman cp -> $name:/app/" -ForegroundColor Cyan
  podman cp (Join-Path $stageDir '.') "$name`:/app/" | Out-Null

  if($InstallDeps){
    # Python deps
    try {
      podman exec $name sh -lc "test -f /app/requirements.txt && python -m pip install --no-cache-dir -r /app/requirements.txt || true" | Out-Null
    } catch {}
    # Node deps
    try {
      podman exec $name sh -lc "test -f /app/package.json && (command -v npm >/dev/null 2>&1 && (npm ci || npm i)) || true" | Out-Null
    } catch {}
  }

  if($Restart){
    Write-Host "     -> restarting $name" -ForegroundColor Yellow
    podman restart $name | Out-Null
  }

  Write-Host " [ok] $name synced" -ForegroundColor Green
}

# --- main ---
if(!(Test-Path $ServicesRoot)){ throw "Services root missing: $ServicesRoot" }
if(!(Get-Command podman -ErrorAction SilentlyContinue)){ throw "Podman not found. Install Podman Desktop." }

New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

foreach($s in $Services){ Sync-One $s }

Write-Host "`nDone. Files vacuumed into running containers (where present)." -ForegroundColor Green
if($InstallDeps){ Write-Host "   Deps were installed for services with requirements.txt / package.json." -ForegroundColor Green }
if($Restart){ Write-Host "   Containers were restarted." -ForegroundColor Green }
