# metatron_vacuum_blocks.ps1  — no-robocopy, quote-safe

param(
  [switch]$InstallDeps,
  [switch]$Restart,
  [switch]$DryRun,
  [string]$ServicesRoot = "C:\Projects\Stacks\nexus-metatron\backend\services"
)

# --- Path resolver (plural cognikubes first) ---
$Here = $PSScriptRoot
$Backend = Split-Path -Parent (Split-Path -Parent $Here)
if (!(Test-Path $Backend)) { $Backend = "C:\Projects\Stacks\nexus-metatron\backend" }
$InfraRoot = Join-Path $Backend "infra\cognikubes"
if (!(Test-Path $InfraRoot)) { $InfraRoot = Join-Path $Backend "infra\cognikube" }
if (-not $PSBoundParameters.ContainsKey("ServicesRoot")) { $ServicesRoot = Join-Path $Backend "services" }
if (!(Test-Path $ServicesRoot)) { throw "Services root missing: $ServicesRoot" }
$ErrorActionPreference = "Stop"

if (!(Get-Command podman -ErrorAction SilentlyContinue)) { throw "Podman not found." }

function Get-ContainerNames { try { (podman ps -a --format "{{.Names}}") -split "`n" | Where-Object { $_ } } catch { @() } }
$script:ALL_CONTAINERS = Get-ContainerNames
function Resolve-ContainerName([string]$base) {
  if (-not $base) { return $null }
  if ($script:ALL_CONTAINERS -contains $base) { return $base }
  $cand = $script:ALL_CONTAINERS | Where-Object { $_ -match "(^|[-_])$([regex]::Escape($base))($|[-_])" -or $_ -like "*$base*" } | Select-Object -First 1
  if ($cand) { return $cand }; return $base
}
function Running([string]$name) {
  try { foreach($r in (podman ps --format "{{.Names}}|{{.Status}}") -split "`n"){ $p = $r -split '\|',2; if($p[0] -eq $name -and $p[1] -match '^Up\b'){ return $true } }; $false } catch { $false }
}

# ---- staging/copy (no robocopy) ----
$ExcludeDirRegex = '(\\|/)(\.git|node_modules|__pycache__|\.venv|venv|\.idea|\.vscode|dist|build|coverage)(\\|/)'
$ExcludeFileExts = @(".pyc",".pyo",".log",".db",".sqlite",".map",".lock")
function New-CleanDir([string]$path){ if(Test-Path $path){ Remove-Item -Recurse -Force $path | Out-Null }; New-Item -ItemType Directory -Force -Path $path | Out-Null }
function Stage-Copy([string]$src, [string]$dst){
  if(!(Test-Path $src)){ throw "Missing source: $src" }
  New-CleanDir $dst
  $srcFull = (Resolve-Path $src).Path
  $files = Get-ChildItem -Path $srcFull -Recurse -File -Force | Where-Object {
    $_.FullName -notmatch $ExcludeDirRegex -and ($ExcludeFileExts -notcontains $_.Extension.ToLower())
  }
  foreach($f in $files){
    $rel = $f.FullName.Substring($srcFull.Length).TrimStart('\','/')
    $outPath = Join-Path $dst $rel
    $outDir = Split-Path $outPath -Parent
    if(!(Test-Path $outDir)){ New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
    Copy-Item -LiteralPath $f.FullName -Destination $outPath -Force
  }
}
function Maybe-InstallDeps($name){
  if(!$InstallDeps){ return }
  try { podman exec $name sh -lc "test -f /app/requirements.txt && python -m pip install --no-cache-dir -r /app/requirements.txt || true" | Out-Null } catch {}
  try { podman exec $name sh -lc "test -f /app/package.json && (command -v npm >/dev/null 2>&1 && (npm ci || npm i)) || true" | Out-Null } catch {}
}
function Sync-DirTo([string]$src, [string]$container, [string]$dest){
  if(!(Test-Path $src)){ Write-Host "   [skip] missing: $src" -ForegroundColor DarkGray; return }
  $resolved = Resolve-ContainerName $container
  if($DryRun){ Write-Host ("   [dry] {0}  ->  {1}" -f $src, ("{0}:{1}" -f $resolved, $dest)) -ForegroundColor Yellow; return }
  if(!(Running $resolved)){ Write-Host "   [skip] container not running: $resolved" -ForegroundColor DarkGray; return }
  $stage = Join-Path $env:TEMP ("stage_"+[IO.Path]::GetFileName($src)+"_"+$resolved)
  Write-Host "   [+] Staging $src" -ForegroundColor Cyan
  Stage-Copy $src $stage
  Write-Host ("      -> podman cp -> {0}" -f ("{0}:{1}" -f $resolved, $dest)) -ForegroundColor Cyan
  podman cp (Join-Path $stage '.') ("{0}:{1}" -f $resolved, $dest) | Out-Null
}

# ---------- BLOCK MAP ----------
$Blocks = @(
  @{ Name="Acidemikube";          Target="acidemikube";        Copies=@(@{From="acidemikube";        To="/app/"}, @{From="berts";               To="/app/berts/"}) },
  @{ Name="AnyNodes";             Target="anynode";            Copies=@(@{From="anynode";            To="/app/"}) },
  @{ Name="Memory";               Target="memory";             Copies=@(@{From="memory";             To="/app/"}); Siblings=@(@{Target="planner"; From="planner"; To="/app/"}) },
  @{ Name="Language Processing";  Target="language-processing";Copies=@(@{From="language-processing"; To="/app/"}) },
  @{ Name="Orchestration";        Target="orchestration";      Copies=@(@{From="orchestration";      To="/app/"}) },
  @{ Name="Master Orchestration"; Target="master-orchestration";Copies=@(@{From="master-orchestration";To="/app/"}) },
  @{ Name="Service Orchestration";Target="service-orchestration";Copies=@(@{From="service-orchestration";To="/app/"}) },
  @{ Name="Consciousness";        Target="consciousness";      Copies=@(@{From="consciousness";      To="/app/"}) },
  @{ Name="Subconsciousness";     Target="subconsciousness";   Copies=@(@{From="subconsciousness";   To="/app/"}) },
  @{ Name="Berts";                Target="berts";              Copies=@(@{From="berts";              To="/app/"}) },
  @{ Name="Visual Cortex";        Target="visual_cortex";      Copies=@(@{From="visual_cortex";      To="/app/"}) },
  @{ Name="Vocal Services";       Target="vocal_services";     Copies=@(@{From="vocal_services";     To="/app/"}) },
  @{ Name="Auditory Services";    Target="auditory_services";  Copies=@(@{From="auditory_services";  To="/app/"}) },
  @{ Name="Archivist";            Target="archivist";          Copies=@(@{From="archivist";          To="/app/"}) },
  @{ Name="Edge ANYNODE";         Target="edge-anynode";       Copies=@(@{From="edge-anynode";       To="/app/"}) },
  @{ Name="LLM Proxy";            Target="llm-proxy";          Copies=@(@{From="llm-proxy";          To="/app/"}) },
  @{ Name="Nexus Core";           Target="nexus-core";         Copies=@(@{From="nexus-core";         To="/app/"}) },
  @{ Name="Viren";                Target="viren";              Copies=@(@{From="viren";              To="/app/"}) }
)

# ---------- main ----------
foreach($b in $Blocks){
  $target = $b.Target
  Write-Host ("`n== Block: {0}  -> container '{1}'" -f $b.Name, $target) -ForegroundColor Magenta

  foreach($c in $b.Copies){
    $src = Join-Path $ServicesRoot $c.From
    Sync-DirTo -src $src -container $target -dest $c.To
    if(!$DryRun){ $resolved = Resolve-ContainerName $target; if(Running $resolved){ Maybe-InstallDeps $resolved } }
  }

  if($b.ContainsKey("Siblings")){
    foreach($s in $b.Siblings){
      Write-Host ("  (sibling) -> {0}" -f $s.Target) -ForegroundColor DarkCyan
      $src = Join-Path $ServicesRoot $s.From
      Sync-DirTo -src $src -container $s.Target -dest $s.To
      if(!$DryRun){ $res = Resolve-ContainerName $s.Target; if(Running $res){ Maybe-InstallDeps $res } }
    }
  }

  if($Restart -and -not $DryRun){
    $toRestart = @($target) + ($b.Siblings | ForEach-Object { $_.Target })
    foreach($t in $toRestart){
      $res = Resolve-ContainerName $t
      if($res -and (Running $res)){
        Write-Host ("   restarting {0}" -f $res) -ForegroundColor Yellow
        podman restart $res | Out-Null
      }
    }
  }
}

Write-Host "`nDone. Vacuum complete." -ForegroundColor Green
if($DryRun){ Write-Host "This was a dry run; no files were copied." -ForegroundColor Yellow }
if($InstallDeps){ Write-Host "Dependencies were installed where configs existed." -ForegroundColor Yellow }
if($Restart){ Write-Host "Containers were restarted after sync." -ForegroundColor Yellow }
