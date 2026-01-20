param(
  [Parameter(Mandatory=$true)][string]$RepoRoot
)

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Pkg = Join-Path $Here "packages"

# Ensure target dirs
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "policy") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "security_shields") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "nexus_scaffold") | Out-Null

# Expand any packages that exist
$packages = Get-ChildItem -File (Join-Path $Pkg "*.zip")
foreach ($p in $packages) {
  Write-Host "Expanding $($p.Name)..."
  $target = switch -Regex ($p.Name) {
    "Policy_Kit" { Join-Path $RepoRoot "policy" }
    "Control_Plane_Kit" { Join-Path $RepoRoot "policy\control_plane_kit" }
    "Security_Shields" { Join-Path $RepoRoot "security_shields" }
    "Scaffold_Stack" { Join-Path $RepoRoot "nexus_scaffold" }
    "CI_Policy_Check" { Join-Path $RepoRoot "" }
    "Category_Canon" { Join-Path $RepoRoot "" }
    default { Join-Path $RepoRoot "extra" }
  }
  Expand-Archive -Force $p.FullName $target
}

# Copy CI/workflows from CI_Policy_Check & Category_Canon if present
$ciRoots = @(
  (Join-Path $RepoRoot ".\Nexus_CI_Policy_Check_v1"),
  (Join-Path $RepoRoot ".\Nexus_Category_Canon_v1")
)
foreach ($root in $ciRoots) {
  if (Test-Path (Join-Path $root ".github")) {
    Copy-Item -Recurse -Force (Join-Path $root ".github") (Join-Path $RepoRoot ".github")
  }
  if (Test-Path (Join-Path $root "ci")) {
    Copy-Item -Recurse -Force (Join-Path $root "ci") (Join-Path $RepoRoot "ci")
  }
}

# Move taxonomy/policy extras into policy/
$catRoot = Join-Path $RepoRoot "Nexus_Category_Canon_v1"
if (Test-Path (Join-Path $catRoot "Nexus_Category_Canon_v1")) {
  $catRoot = Join-Path $catRoot "Nexus_Category_Canon_v1"
}
if (Test-Path (Join-Path $catRoot "TAXONOMY.md")) {
  Copy-Item -Force (Join-Path $catRoot "TAXONOMY.md") (Join-Path $RepoRoot "TAXONOMY.md")
}
if (Test-Path (Join-Path $catRoot "discovery.cap_register.schema.json")) {
  Copy-Item -Force (Join-Path $catRoot "discovery.cap_register.schema.json") (Join-Path $RepoRoot "policy\discovery.cap_register.schema.json")
}
if (Test-Path (Join-Path $catRoot "mapping.services.json")) {
  Copy-Item -Force (Join-Path $catRoot "mapping.services.json") (Join-Path $RepoRoot "policy\mapping.services.json")
}

Write-Host "Install complete. Next: start NATS and follow each kit's README."
