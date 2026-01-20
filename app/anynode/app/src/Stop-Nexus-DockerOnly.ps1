<#
 Stop-Nexus-DockerOnly.ps1
 Stops and removes the stack using docker compose only.
#>
param([string]$Root = ".\Nexus_Merged_Pack")
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker CLI not found." }
$ComposeFile = Join-Path $Root "compose.yaml"
if (-not (Test-Path $ComposeFile)) { throw "Compose file not found: $ComposeFile" }
Push-Location $Root
try { docker compose -f "$ComposeFile" down } finally { Pop-Location }
