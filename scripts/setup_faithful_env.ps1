param(
    [string]$EnvName = "scenethesis-faithful",
    [string]$PythonVersion = "3.11",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [string]$TorchVersion = "2.5.1",
    [string]$TorchvisionVersion = "0.20.1",
    [string]$CudaToolkitVersion = "12.4.1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReposDir = Join-Path $RepoRoot "models\repos"
New-Item -ItemType Directory -Force -Path $ReposDir | Out-Null

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Command,
        [string[]]$Arguments = @()
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command $($Arguments -join ' ')"
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found. Install Miniconda/Anaconda or run this from a shell where conda is available."
}

$CondaBase = (& conda info --base).Trim()
$EnvPath = Join-Path $CondaBase "envs\$EnvName"
if (-not (Test-Path $EnvPath)) {
    Invoke-Checked -Command conda -Arguments @("create", "-y", "-n", $EnvName, "python=$PythonVersion")
}
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82")
Invoke-Checked -Command conda -Arguments @("install", "-y", "-n", $EnvName, "--override-channels", "-c", "nvidia/label/cuda-12.4.1", "-c", "defaults", "cuda-toolkit=$CudaToolkitVersion", "cuda-nvcc=12.4.*", "cuda-version=12.4")
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "torch==$TorchVersion", "torchvision==$TorchvisionVersion", "--index-url", $TorchIndexUrl)
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "openai", "pydantic", "PyYAML", "python-dotenv", "numpy<2", "pillow", "opencv-python<4.13", "opencv-python-headless<4.13", "matplotlib", "pycocotools", "trimesh", "rtree", "transformers==4.37.2", "open_clip_torch", "romatch")
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "--no-cache-dir", "numpy<2")
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git")

$GroundingDir = Join-Path $ReposDir "GroundingDINO"
if (-not (Test-Path $GroundingDir)) {
    Invoke-Checked -Command git -Arguments @("clone", "https://github.com/IDEA-Research/GroundingDINO.git", $GroundingDir)
}
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "--no-build-isolation", "-e", $GroundingDir)
Invoke-Checked -Command powershell -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "install_groundingdino_windows.ps1"), "-EnvName", $EnvName)

$DepthProDir = Join-Path $ReposDir "ml-depth-pro"
if (-not (Test-Path $DepthProDir)) {
    Invoke-Checked -Command git -Arguments @("clone", "https://github.com/apple/ml-depth-pro.git", $DepthProDir)
}
Invoke-Checked -Command conda -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-e", $DepthProDir)

Write-Host "Attempting PyTorch3D install through scripts/install_pytorch3d_windows.ps1."
Invoke-Checked -Command powershell -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "install_pytorch3d_windows.ps1"), "-EnvName", $EnvName)

Write-Host "Faithful environment setup complete: $EnvName"
Write-Host "Next: conda run -n $EnvName python scripts/download_faithful_checkpoints.py"
