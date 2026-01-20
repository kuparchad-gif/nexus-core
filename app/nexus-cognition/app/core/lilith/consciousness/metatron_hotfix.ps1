$path="C:\Projects\Stacks\nexus-metatron\backend\infra\cognikubes\metatron_hotfix.ps1"
@'
param(
  [string]$ServicesRoot = "C:\Projects\Stacks\nexus-metatron\backend\services"
)

# Path resolver (plural first)
$Here = $PSScriptRoot
$Backend = Split-Path -Parent (Split-Path -Parent $Here)
if (!(Test-Path $Backend)) { $Backend = "C:\Projects\Stacks\nexus-metatron\backend" }
if (-not $PSBoundParameters.ContainsKey("ServicesRoot")) { $ServicesRoot = Join-Path $Backend "services" }
$ErrorActionPreference = "Stop"

function Replace-In-File($path,$pattern,$replacement){
  if(!(Test-Path $path)){ return $false }
  $txt = Get-Content $path -Raw
  $new = [regex]::Replace($txt,$pattern,$replacement)
  if($new -ne $txt){ Set-Content -Path $path -Value $new -Encoding UTF8; return $true }
  return $false
}

# 1) Upgrade Ray pins across services
$reqs = Get-ChildItem $ServicesRoot -Recurse -Filter requirements.txt -ErrorAction SilentlyContinue
foreach($r in $reqs){ Replace-In-File $r.FullName 'ray(\[default\])?==\d+\.\d+\.\d+' 'ray[default]==2.48.0' | Out-Null }

# 2) Poetry flag modernization
$dockerfiles = Get-ChildItem $ServicesRoot -Recurse -Filter Dockerfile -ErrorAction SilentlyContinue
foreach($df in $dockerfiles){ Replace-In-File $df.FullName '--no-dev' '--without dev' | Out-Null }

# 3) Fix viren COPY if present
$VIREN = Join-Path $ServicesRoot 'viren\Dockerfile'
if(Test-Path $VIREN){ Replace-In-File $VIREN 'COPY\s+services/viren\s+/app' 'COPY . /app' | Out-Null }

# 4) Refresh ray-head to py312-compatible ray
try { podman rm -f ray-head | Out-Null } catch {}
podman run -d --pod metatron --name ray-head docker.io/rayproject/ray:2.48.0-py312 `
  bash -lc "ray start --head --dashboard-host 0.0.0.0" | Out-Null

Write-Host "Hotfix applied: ray pins, poetry flags, viren COPY, ray-head refreshed." -ForegroundColor Green
'@ | Set-Content -Encoding UTF8 -Path $path
