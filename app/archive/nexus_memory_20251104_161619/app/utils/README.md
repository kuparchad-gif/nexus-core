# Nexus Full Package (v1a)
Generated: 2025-09-01 08:07:29

Included kits:


MISSING kits (not found on disk):
- Nexus_Canon_Policy_Kit_v1p1.zip
- Nexus_Control_Plane_Kit_v1.zip
- Nexus_Security_Shields_v1.zip
- Nexus_Scaffold_Stack_v1.zip
- Nexus_CI_Policy_Check_v1.zip
- Nexus_Category_Canon_v1.zip

## Install (PowerShell 7+)
```powershell
$RepoRoot = "C:\\Projects\\Stacks\\nexus-metatron"
Expand-Archive -Force "$PWD\\Nexus_Full_Package_v1a.zip" "$RepoRoot\\Nexus_Full_Package"
& "$RepoRoot\\Nexus_Full_Package\\install_all.ps1" -RepoRoot $RepoRoot
```
