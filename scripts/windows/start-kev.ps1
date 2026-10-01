$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$env:KEV_DTYPE = 'fp32'
$env:KEV_MERGE = '0'
$env:KEV_CUDA_GRAPHS = '0'
$env:KEV_FUSED = '0'
$env:KEV_ATTN = 'eager'
$lock = Get-Content configs/kev-resolved.json -Raw | ConvertFrom-Json
& .venv-kev/Scripts/python.exe scripts/check_gpu.py --out handoff/environment-kev.json
if ($LASTEXITCODE -ne 0) { throw 'Kev CUDA check failed' }
& .venv-kev/Scripts/python.exe -u -m kev.serve --run ($lock.model + '@' + $lock.revision) --host 127.0.0.1 --port 8009
if ($LASTEXITCODE -ne 0) { throw 'Kev server failed' }
