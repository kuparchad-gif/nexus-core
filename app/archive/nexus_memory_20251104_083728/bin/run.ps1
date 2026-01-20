
param(
  [string]$SrcRoot = "C:\Projects\Stacks\nexus-metatron\CogniKubes",
  [string]$DestRoot = "C:\Projects\Stacks\nexus-metatron\CogniKubes.organized",
  [string]$Provider = "lmstudio",
  [string]$LmStudioUrl = "http://192.168.162:1234/v1/chat/completions",
  [string]$LmStudioModel = "qwen/qwen3-4b-thinking-2507",
  [string]$SupportMode = "tree",
  [string]$DryRun = "false",
  [double]$ConfidenceMin = 0.4,
  [bool]$AllowSamePath = $false
)

$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv")) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:PROVIDER = $Provider
$env:LMSTUDIO_URL = $LmStudioUrl
$env:LMSTUDIO_MODEL = $LmStudioModel
$env:SRC_ROOT = $SrcRoot
$env:DEST_ROOT = $DestRoot
$env:DO_METADATA = "true"
$env:SUPPORT_MODE = $SupportMode
$env:CONFIDENCE_MIN = "$ConfidenceMin"
$env:ALLOW_SAME_PATH = ($AllowSamePath ? "true" : "false")
$env:DRY_RUN = $DryRun

python .\apps\main.py
