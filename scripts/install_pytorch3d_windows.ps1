param(
    [string]$EnvName = "scenethesis-faithful",
    [string]$CudaToolkitPath = "",
    [string]$CudaArchList = "8.9",
    [string]$BuildTemp = "C:\stmp"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

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
$PythonExe = Join-Path $EnvPath "python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Conda environment Python was not found: $PythonExe"
}

if (-not $CudaToolkitPath) {
    $EnvCuda = $EnvPath
    $EnvNvcc = Join-Path $EnvCuda "bin\nvcc.exe"
    $EnvLibraryNvcc = Join-Path $EnvCuda "Library\bin\nvcc.exe"
    if ((Test-Path $EnvNvcc) -or (Test-Path $EnvLibraryNvcc)) {
        $CudaToolkitPath = $EnvCuda
    } else {
        $SystemCuda124 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
        if (Test-Path (Join-Path $SystemCuda124 "bin\nvcc.exe")) {
            $CudaToolkitPath = $SystemCuda124
        }
    }
}
if (-not $CudaToolkitPath) {
    throw "CUDA Toolkit 12.4 with nvcc was not found. Install it with: conda install -n $EnvName -c nvidia cuda-toolkit=12.4.1"
}
$CudaToolkitPath = (Resolve-Path $CudaToolkitPath).Path
$Nvcc = Join-Path $CudaToolkitPath "bin\nvcc.exe"
$LibraryNvcc = Join-Path $CudaToolkitPath "Library\bin\nvcc.exe"
if (-not (Test-Path $Nvcc) -and (Test-Path $LibraryNvcc)) {
    $Nvcc = $LibraryNvcc
}
if (-not (Test-Path $Nvcc)) {
    throw "nvcc was not found under CUDA toolkit path: $Nvcc"
}

$CubHome = Join-Path $CudaToolkitPath "include"
if (-not (Test-Path (Join-Path $CubHome "cub\cub.cuh"))) {
    $CubDir = Join-Path $RepoRoot "models\repos\cub"
    if (-not (Test-Path $CubDir)) {
        Invoke-Checked -Command git -Arguments @("clone", "https://github.com/NVIDIA/cub.git", $CubDir)
    }
    $CubHome = (Resolve-Path $CubDir).Path
}

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe was not found. Install Visual Studio 2022 Build Tools with the C++ workload."
}
$VsInstall = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $VsInstall) {
    throw "Visual Studio C++ Build Tools were not found."
}
$VsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $VsDevCmd)) {
    throw "VsDevCmd.bat was not found: $VsDevCmd"
}
New-Item -ItemType Directory -Force -Path $BuildTemp | Out-Null
$BuildTemp = (Resolve-Path $BuildTemp).Path

$InstallCommand = "call `"$VsDevCmd`" -arch=x64 -host_arch=x64 && " +
    "set `"DISTUTILS_USE_SDK=1`" && " +
    "set `"MSSdk=1`" && " +
    "set `"CUDA_HOME=$CudaToolkitPath`" && " +
    "set `"CUDA_PATH=$CudaToolkitPath`" && " +
    "set `"CUB_HOME=$CubHome`" && " +
    "set `"TORCH_CUDA_ARCH_LIST=$CudaArchList`" && " +
    "set `"MAX_JOBS=1`" && " +
    "set `"TMP=$BuildTemp`" && " +
    "set `"TEMP=$BuildTemp`" && " +
    "set `"PATH=$EnvPath\Scripts;$EnvPath\bin;$EnvPath\Library\bin;$CudaToolkitPath\bin;$CudaToolkitPath\Library\bin;$CudaToolkitPath\libnvvp;!PATH!`" && " +
    "where cl && where ninja && where nvcc && " +
    "`"$PythonExe`" -m pip install --no-build-isolation --no-cache-dir git+https://github.com/facebookresearch/pytorch3d.git"

cmd /v:on /c $InstallCommand
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch3D build failed with exit code $LASTEXITCODE."
}

Invoke-Checked -Command $PythonExe -Arguments @("-c", "import pytorch3d, torch; print('pytorch3d ok'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())")
