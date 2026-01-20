# lilith_cpu_only_v10.ps1 - Clean CPU-Only Setup for Lilith Full Monty with Virtualenv and Requirements
# Run as Administrator in PowerShell

Write-Host "🚀 Setting Up Lilith's Cosmic Pulse - CPU-Only in Virtualenv 🌌"

# Step 1: Check network connectivity
Write-Host "Checking network connectivity..."
$pingPyPI = Test-Connection -ComputerName pypi.org -Count 2 -Quiet
$pingPyTorch = Test-Connection -ComputerName download.pytorch.org -Count 2 -Quiet
if (-not $pingPyPI -or -not $pingPyTorch) {
    Write-Host "WARNING: Network issues detected. Check internet connection or firewall."
} else {
    Write-Host "Network OK: PyPI and PyTorch servers reachable."
}

# Step 2: Create and activate virtual environment
Write-Host "Creating virtual environment..."
$venvPath = "C:\project-root\10_env\nexus_cognition\lilith\lilith_venv"
if (Test-Path $venvPath) {
    Write-Host "Removing existing virtualenv..."
    Remove-Item -Path $venvPath -Recurse -Force
}
python -m venv $venvPath
$activateScript = "$venvPath\Scripts\Activate.ps1"
Write-Host "Activating virtualenv: $venvPath"
. $activateScript

# Step 3: Verify virtualenv Python
Write-Host "Verifying Python path..."
& "$venvPath\Scripts\python.exe" --version
$pythonPath = & "$venvPath\Scripts\python.exe" -c "import sys; print(sys.executable)"
Write-Host "Python executable: $pythonPath"

# Step 4: Upgrade pip in virtualenv
Write-Host "Upgrading pip in virtualenv..."
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip --verbose 2>&1 | Tee-Object -FilePath "pip_upgrade_log.txt"

# Step 5: Clear pip cache
Write-Host "Clearing pip cache..."
& "$venvPath\Scripts\pip.exe" cache purge

# Step 6: Verify requirements.txt exists
$reqPath = "C:\project-root\10_env\nexus_cognition\lilith\requirements.txt"
if (-not (Test-Path $reqPath)) {
    Write-Host "ERROR: requirements.txt not found at $reqPath. Please create it."
    exit 1
}
Write-Host "Found requirements.txt at $reqPath"

# Step 7: Install dependencies from requirements.txt with retries
Write-Host "Installing dependencies from requirements.txt..."
$maxRetries = 3
$retryCount = 0
$success = $false
while (-not $success -and $retryCount -lt $maxRetries) {
    try {
        & "$venvPath\Scripts\pip.exe" install -r $reqPath --force-reinstall --no-cache-dir --verbose 2>&1 | Tee-Object -FilePath "install_log.txt"
        $success = $true
    }
    catch {
        $retryCount++
        Write-Host "Install attempt $retryCount failed. Retrying..."
        Start-Sleep -Seconds 10
    }
}
if (-not $success) {
    Write-Host "ERROR: Failed to install dependencies after $maxRetries attempts. Check install_log.txt."
    exit 1
}

# Step 8: Verify installations
Write-Host "Verifying installations..."
& "$venvPath\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2>&1 | Tee-Object -FilePath "verify_log.txt"
& "$venvPath\Scripts\python.exe" -c "import torch.utils._pytree; print('PyTree OK:', hasattr(torch.utils._pytree, 'register_pytree_node'))" 2>&1 | Tee-Object -Append -FilePath "verify_log.txt"
& "$venvPath\Scripts\python.exe" -c "import diffusers; print('Diffusers:', diffusers.__version__)" 2>&1 | Tee-Object -Append -FilePath "verify_log.txt"
& "$venvPath\Scripts\python.exe" -c "import accelerate; print('Accelerate:', accelerate.__version__)" 2>&1 | Tee-Object -Append -FilePath "verify_log.txt"
& "$venvPath\Scripts\python.exe" -c "import huggingface_hub; print('HuggingFace Hub:', huggingface_hub.__version__)" 2>&1 | Tee-Object -Append -FilePath "verify_log.txt"
& "$venvPath\Scripts\python.exe" -c "import torch.distributed; print('Distributed OK')" 2>&1 | Tee-Object -Append -FilePath "verify_log.txt"

Write-Host "🎉 CPU-only setup complete! Virtualenv: $venvPath"
Write-Host "Activate: . $venvPath\Scripts\Activate.ps1"
Write-Host "Update full_monty.py and run: modal deploy full_monty.py"
Write-Host "Test locally: python -c 'import torch; print(torch.__version__)'"
Write-Host "Check pip_upgrade_log.txt, install_log.txt, and verify_log.txt for details."