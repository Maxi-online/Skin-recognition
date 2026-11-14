[CmdletBinding()]
param(
    [ValidateSet('v42', 'v4', 'v3', 'v2')]
    [string]$Backend = 'v42',
    [ValidateSet('hybrid', 'skin_texture', 'whole_face', 'whole_face_detector_only')]
    [string]$Method = 'hybrid',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [string]$FreezePath,
    [string]$DatabasePath,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$environmentName = if ($Backend -in @('v4', 'v42')) { '.venv-gpu' } else { '.venv' }
$pythonPath = Join-Path $projectRoot "$environmentName/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Required Python environment is missing: $pythonPath"
}
if (-not $FreezePath) {
    $profilePath = Join-Path $projectRoot 'config/research_profile.json'
    $profile = if (Test-Path -LiteralPath $profilePath) { Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json } else { $null }
    if ($profile -and $profile.backend -eq $Backend -and $profile.freeze_path) {
        $FreezePath = $profile.freeze_path
    }
    else {
        $freezeName = if ($Backend -eq 'v42') { 'study_v4_2.json' } else { 'study_v4_1.json' }
        $FreezePath = Join-Path $projectRoot "research/frozen/$freezeName"
    }
}
if (-not $DatabasePath) {
    $galleryName = if ($Backend -in @('v4', 'v42')) { "skin_embeddings_${Backend}_$Method" } else { "skin_embeddings_$Backend" }
    $DatabasePath = Join-Path $projectRoot "runtime/$galleryName"
}
if (-not [System.IO.Path]::IsPathRooted($FreezePath)) {
    $FreezePath = Join-Path $projectRoot $FreezePath
}
if (-not [System.IO.Path]::IsPathRooted($DatabasePath)) {
    $DatabasePath = Join-Path $projectRoot $DatabasePath
}
$env:SKIN_BACKEND = $Backend
$env:DATABASE_PATH = $DatabasePath
$env:HOST = '127.0.0.1'
$env:PORT = [string]$Port
$env:WORKERS = '1'
$env:PYTHONUTF8 = '1'
$env:NO_ALBUMENTATIONS_UPDATE = '1'
if ($Backend -in @('v4', 'v42')) {
    if (-not (Test-Path -LiteralPath $FreezePath -PathType Leaf)) {
        throw "Frozen v4 profile is missing: $FreezePath"
    }
    $env:SKIN_V4_FREEZE_PATH = $FreezePath
    $env:SKIN_V4_METHOD = $Method
    $env:SKIN_V4_DATABASE_PATH = $DatabasePath
    & $pythonPath -c "import torch; import onnxruntime as ort; assert torch.cuda.is_available(), 'Frozen V4 profile requires CUDA'; assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA ONNX Runtime is required'"
    if ($LASTEXITCODE -ne 0) { throw 'The frozen CUDA runtime is unavailable.' }
}
elseif ($Backend -eq 'v3') {
    $env:SKIN_V3_DATABASE_PATH = $DatabasePath
}

[PSCustomObject]@{
    Backend = $Backend
    Method = if ($Backend -in @('v4', 'v42')) { $Method } else { 'configured texture profile' }
    Python = $pythonPath
    Url = "http://127.0.0.1:$Port/"
    Database = $DatabasePath
    Workers = 1
} | Format-List

if ($CheckOnly) { return }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath (Join-Path $projectRoot 'run_service.py')
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
