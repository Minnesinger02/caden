$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "../..")
$env:PYTHONUTF8 = "1"
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.tools/python'
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
$useUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
if (!(Test-Path ".venv-win/Scripts/python.exe")) {
    if ($useUv) { uv venv --python 3.12 .venv-win }
    else { py -3.12 -m venv .venv-win }
    Check-Exit
}
$python = ".venv-win/Scripts/python.exe"
if ($useUv) {
    uv pip install --python $python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    Check-Exit
    uv pip install --python $python -r scripts/windows/requirements.txt
    Check-Exit
    uv pip check --python $python
} else {
    & $python -m pip install --upgrade pip
    Check-Exit
    & $python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    Check-Exit
    & $python -m pip install -r scripts/windows/requirements.txt
    Check-Exit
    & $python -m pip check
}
Check-Exit
& $python scripts/check_gpu.py --out environment-windows.json
Check-Exit
& $python -m unittest discover -s tests -v
Check-Exit
Write-Host "Ready. Read START_HERE_WINDOWS.md."
