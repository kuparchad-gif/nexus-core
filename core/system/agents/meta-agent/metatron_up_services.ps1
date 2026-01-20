param(
  [switch]$Rebuild,
  [switch]$Recreate,
  [string]$ServicesRoot = "C:\Projects\Stacks\nexus-metatron\backend\services"
)

$ErrorActionPreference = "Stop"
function Exec($c){ Write-Host ">> $c" -ForegroundColor Cyan; iex $c; if($LASTEXITCODE -ne 0){ throw "Command failed: $c (exit $LASTEXITCODE)" } }
if (!(Get-Command podman -ErrorAction SilentlyContinue)) { throw "Podman not found. Install Podman Desktop." }

# pod detection/creation
function Get-PodFrom([string[]]$names){ foreach($n in $names){ try { $p = podman inspect $n --format "{{.PodName}}" 2>$null; if($p -and $p -ne "<no value>"){ return $p } } catch {} }; return $null }
$pod = Get-PodFrom @("qdrant","loki","ray-head")
if (-not $pod) {
  $pod = "metatron"
  $ports = @(
    "-p 1313:1313","-p 8011:8011","-p 8008:8008","-p 8003:8003",
    "-p 8010:8010","-p 8007:8007","-p 8009:8009","-p 8012:8012",
    "-p 8013:8013","-p 8014:8014","-p 8020:8020",
    "-p 6333:6333","-p 3100:3100","-p 8265:8265","-p 10001:10001","-p 6379:6379"
  ) -join " "
  if (!(podman pod exists $pod)) { Exec "podman pod create --name $pod $ports" }
} else {
  Write-Host "Using existing pod: $pod" -ForegroundColor Green
}

function Has-Dockerfile($dir){ Test-Path (Join-Path $dir "Dockerfile") -or Test-Path (Join-Path $dir "Dockerfile.nova") }
function Dockerfile-Path($dir){ if(Test-Path (Join-Path $dir "Dockerfile.nova")){ Join-Path $dir "Dockerfile.nova" } else { Join-Path $dir "Dockerfile" } }
function Img-Tag([string]$name){ $name.ToLower() }
function C-Exists([string]$name){ try { (podman ps -a --format "{{.Names}}") -split "`n" -contains $name } catch { $false } }
function C-Running([string]$name){ try { foreach($r in (podman ps --format "{{.Names}}|{{.Status}}") -split "`n"){ $p = $r -split '\|',2; if($p[0] -eq $name -and $p[1] -match '^Up\b'){ return $true } }; $false } catch { $false } }

function BuildRun($svc){
  $name = $svc.Name
  $dir  = Join-Path $ServicesRoot $svc.Dir
  if (!(Test-Path $dir)) { Write-Host ("[ ] {0} (missing dir)" -f $name) -ForegroundColor DarkGray; return }
  if (!(Has-Dockerfile $dir)) { Write-Host ("[ ] {0} (no Dockerfile in {1})" -f $name,$dir) -ForegroundColor DarkGray; return }
  $df = Dockerfile-Path $dir
  $img = Img-Tag $name

  Write-Host ("[+] {0}" -f $name) -ForegroundColor Cyan
  Exec ("podman build -t {0} -f `"{1}`" `"{2}`"" -f $img, $df, $dir)

  if (C-Exists $name) {
    if ($Recreate) { if (C-Running $name) { Exec "podman stop $name" }; Exec "podman rm $name" }
    elseif (C-Running $name) { Write-Host "    running -> skip run" -ForegroundColor DarkGray; return }
    else { Exec "podman rm $name" }
  }

  $envArgs = @()
  foreach($k in $svc.Env.Keys){ $envArgs += @("-e","$k=$($svc.Env[$k])") }

  $run = @("podman","run","-d","--pod",$pod,"--name",$name) + $envArgs + @($img)
  Exec ($run -join " ")
}

$Services = @(
  @{ Name="acidemikube";        Dir="acidemikube";        Env=@{ SERVICE_NAME="acidemikube"; QDRANT_URL="http://qdrant:6333"; ANYNODE_HUB_URL="http://anynode:8012" } },
  @{ Name="anynode";            Dir="anynode";            Env=@{ SERVICE_NAME="anynode" } },
  @{ Name="memory";             Dir="memory";             Env=@{ SERVICE_NAME="memory"; QDRANT_URL="http://qdrant:6333"; COLLECTION="metatron" } },
  @{ Name="planner";            Dir="planner";            Env=@{ RAY_ADDRESS="auto"; MEMORY_URL="http://memory:8003"; LLM_OPENAI_BASE="http://llm-proxy:8008/v1"; LLM_MODEL="qwen2.5-3b-instruct" } },
  @{ Name="llm-proxy";          Dir="llm-proxy";          Env=@{ OLLAMA_URL="http://host.docker.internal:11434"; LMSTUDIO_URL="http://host.docker.internal:1234/v1"; DEFAULT_MODEL="qwen2.5:3b" } },
  @{ Name="language-processing";Dir="language-processing";Env=@{ SERVICE_NAME="langproc"; LLM_BASE_URL="http://llm-proxy:8008/v1" } },
  @{ Name="visual_cortex";      Dir="visual_cortex";      Env=@{ SERVICE_NAME="visual_cortex"; LLM_BASE_URL="http://llm-proxy:8008/v1" } },
  @{ Name="vocal_services";     Dir="vocal_services";     Env=@{ SERVICE_NAME="vocal_services" } },
  @{ Name="auditory_services";  Dir="auditory_services";  Env=@{ SERVICE_NAME="auditory_services" } },
  @{ Name="archivist";          Dir="archivist";          Env=@{ SERVICE_NAME="archivist"; QDRANT_URL="http://qdrant:6333"; COLLECTION="metatron"; LOKI_URL="http://loki:3100" } },
  @{ Name="edge-anynode";       Dir="edge-anynode";       Env=@{ TARGET_PLANNER_URL="http://planner:8011" } },
  @{ Name="consciousness";      Dir="consciousness";      Env=@{ SERVICE_NAME="consciousness"; PLANNER_URL="http://planner:8011"; MEMORY_URL="http://memory:8003"; LLM_BASE_URL="http://llm-proxy:8008/v1" } },
  @{ Name="subconsciousness";   Dir="subconsciousness";   Env=@{ SERVICE_NAME="subconsciousness"; MEMORY_URL="http://memory:8003"; LLM_OPENAI_BASE="http://llm-proxy:8008/v1"; LLM_MODEL="qwen2.5-3b-instruct" } },
  @{ Name="nexus-core";         Dir="nexus-core";         Env=@{ SERVICE_NAME="nexus-core"; PLANNER_URL="http://planner:8011"; LOKI_URL="http://loki:3100" } },
  @{ Name="viren";              Dir="viren";              Env=@{ SERVICE_NAME="viren"; LOKI_URL="http://loki:3100" } }
)

foreach($svc in $Services){ BuildRun $svc }
Write-Host "`n=== Status ===" -ForegroundColor Green; Exec "podman pod ps"; Exec "podman ps --pod"
Write-Host "`nTip: now run your vacuum: .\metatron_vacuum_blocks.ps1 -InstallDeps -Restart" -ForegroundColor Yellow
