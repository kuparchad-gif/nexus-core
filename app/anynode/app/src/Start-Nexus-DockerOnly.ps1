<#
 Start-Nexus-DockerOnly.ps1
 Launches Nexus_Merged_Pack using **docker compose only**.
 - Builds images (optional), then starts the stack
 - Performs basic health checks (NATS, Router)
#>

param(
  [string]$Root = ".\Nexus_Merged_Pack",
  [switch]$Rebuild
)

function Require-DockerCompose {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI not found. Install Docker Desktop or Docker Engine."
  }
  try {
    docker compose version | Out-Null
  } catch {
    throw "Your Docker CLI doesn't support 'docker compose'. Please upgrade Docker."
  }
}

Require-DockerCompose

$ComposeFile = Join-Path $Root "compose.yaml"
if (-not (Test-Path $ComposeFile)) { throw "Compose file not found: $ComposeFile" }

# Prefer .env.pinned > .env
$EnvFile = if (Test-Path (Join-Path $Root ".env.pinned")) { ".env.pinned" }
           elseif (Test-Path (Join-Path $Root ".env"))    { ".env" }
           else { $null }
if ($EnvFile) { Write-Host ">> Using env file: $EnvFile" -ForegroundColor DarkCyan }

Push-Location $Root
try {
  if ($Rebuild) {
    Write-Host ">> Building images (parallel)..." -ForegroundColor Cyan
    if ($EnvFile) { docker compose -f "$ComposeFile" --env-file "$EnvFile" build --parallel }
    else { docker compose -f "$ComposeFile" build --parallel }
  } else {
    Write-Host ">> Skipping rebuild. Use -Rebuild to force." -ForegroundColor DarkGray
  }

  Write-Host ">> Starting services with docker compose..." -ForegroundColor Cyan
  if ($EnvFile) { docker compose -f "$ComposeFile" --env-file "$EnvFile" up -d }
  else { docker compose -f "$ComposeFile" up -d }

  # Basic health checks
  Write-Host ">> Waiting for NATS (4222)..." -ForegroundColor Cyan
  $ok = $false; 1..20 | ForEach-Object {
    try { 
      $client = New-Object System.Net.Sockets.TcpClient
      $iar = $client.BeginConnect('127.0.0.1', 4222, $null, $null)
      $wait = $iar.AsyncWaitHandle.WaitOne(500)
      if ($wait -and $client.Connected) { $ok = $true; $client.Close(); break }
      $client.Close()
    } catch {}
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { Write-Warning "NATS not reachable on 4222 yet." } else { Write-Host "NATS OK" -ForegroundColor Green }

  Write-Host ">> Checking Router health (http://localhost:8088/healthz)..." -ForegroundColor Cyan
  $routerOk = $false
  1..30 | ForEach-Object {
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8088/healthz" -TimeoutSec 2
      if ($resp.StatusCode -eq 200) { $routerOk = $true; break }
    } catch {}
    Start-Sleep -Milliseconds 500
  }
  if ($routerOk) { Write-Host "Router OK" -ForegroundColor Green } else { Write-Warning "Router health endpoint not responding (yet)." }

  Write-Host ">> Docker status:" -ForegroundColor Cyan
  docker ps
  Write-Host ">> Logs hint: docker logs uce-router -f" -ForegroundColor DarkGray
}
finally {
  Pop-Location
}
