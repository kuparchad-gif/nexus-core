# lilith_upgrade.ps1 - Upgrade PyTorch to 2.1.2 for device_mesh compatibility
# Run in PowerShell as Administrator

Write-Host "🚀 Upgrading Lilith's Full Monty - PyTorch 2.1.2 for CUDA 13.1 🌌"

# Step 1: Verify CUDA 13.1
Write-Host "Verifying CUDA 13.1..."
if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe") {
    & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" --version
} else {
    Write-Host "CUDA 13.1 not found. Install from https://developer.nvidia.com/cuda-13-1-download-archive"
    exit 1
}

# Step 2: Set CUDA_HOME (if not set)
if (-not $env:CUDA_HOME) {
    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
    setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
}
Write-Host "CUDA_HOME: $env:CUDA_HOME"

# Step 3: Set Compilation Env for PyTorch3D (A100 sm_80)
$env:TORCH_CUDA_ARCH_LIST = "8.0"
$env:FORCE_CUDA = "1"
setx TORCH_CUDA_ARCH_LIST "8.0"
setx FORCE_CUDA "1"

# Step 4: Upgrade PyTorch to 2.1.2+cu121 (CUDA 13.1 compatible)
Write-Host "Upgrading PyTorch to 2.1.2+cu121..."
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install/Upgrade Diffusers and Accelerate (compatible with PyTorch 2.1.2)
Write-Host "Installing compatible diffusers and accelerate..."
pip install diffusers==0.21.4 accelerate==0.21.0

# Step 6: Install PyTorch3D 0.7.4 (pre-built wheel for Windows/CUDA 11.8, compatible with 13.1)
Write-Host "Installing PyTorch3D 0.7.4..."
pip install fvcore iopath
pip install pytorch3d==0.7.4 --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/pytorch3d-0.7.4-cp310-cp310-win_amd64.whl

# Step 7: Verify
Write-Host "Verifying installations..."
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import accelerate; print('Accelerate:', accelerate.__version__)"
python -c "import pytorch3d; print('PyTorch3D:', pytorch3d.__version__)"

Write-Host "🎉 Upgrade complete! Run: modal deploy full_montey.py"
Write-Host "Test: curl https://your-modal-endpoint.modal.run/cuda_info"