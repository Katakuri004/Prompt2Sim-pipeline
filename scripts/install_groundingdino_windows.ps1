param(
    [string]$EnvName = "scenethesis-faithful",
    [string]$CudaToolkitPath = "",
    [string]$CudaArchList = "8.9",
    [string]$BuildTemp = "C:\stmp"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$GroundingDir = Join-Path $RepoRoot "models\repos\GroundingDINO"

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

if (-not (Test-Path $GroundingDir)) {
    throw "GroundingDINO repo was not found: $GroundingDir"
}

$CondaBase = (& conda info --base).Trim()
$EnvPath = Join-Path $CondaBase "envs\$EnvName"
$PythonExe = Join-Path $EnvPath "python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Conda environment Python was not found: $PythonExe"
}

if (-not $CudaToolkitPath) {
    $CudaToolkitPath = $EnvPath
}
$CudaToolkitPath = (Resolve-Path $CudaToolkitPath).Path
if (-not (Test-Path (Join-Path $CudaToolkitPath "bin\nvcc.exe"))) {
    throw "nvcc was not found under CUDA toolkit path: $CudaToolkitPath"
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

$BuildCommand = "call `"$VsDevCmd`" -arch=x64 -host_arch=x64 && " +
    "set `"DISTUTILS_USE_SDK=1`" && " +
    "set `"MSSdk=1`" && " +
    "set `"CUDA_HOME=$CudaToolkitPath`" && " +
    "set `"CUDA_PATH=$CudaToolkitPath`" && " +
    "set `"TORCH_CUDA_ARCH_LIST=$CudaArchList`" && " +
    "set `"MAX_JOBS=1`" && " +
    "set `"TMP=$BuildTemp`" && " +
    "set `"TEMP=$BuildTemp`" && " +
    "set `"PATH=$EnvPath\Scripts;$EnvPath\bin;$EnvPath\Library\bin;$CudaToolkitPath\bin;!PATH!`" && " +
    "cd /d `"$GroundingDir`" && " +
    "`"$PythonExe`" setup.py build_ext --inplace"

cmd /v:on /c $BuildCommand
if ($LASTEXITCODE -ne 0) {
    throw "GroundingDINO native extension build failed with exit code $LASTEXITCODE."
}

Invoke-Checked -Command $PythonExe -Arguments @("-c", "import torch; import groundingdino._C; print('groundingdino extension ok')")
