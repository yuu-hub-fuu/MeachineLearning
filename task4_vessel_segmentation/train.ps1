$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
python -u -X utf8 pipeline.py train --steps 1800 --eval-every 150 2>&1 | Tee-Object -FilePath runs/train.log
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect runs/train.log.' }
python -u -X utf8 predict_deterministic.py
if ($LASTEXITCODE -ne 0) { throw 'Deterministic inference failed.' }
