$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -u -X utf8 predict_deterministic.py
if ($LASTEXITCODE -ne 0) { throw 'Inference failed.' }
python -X utf8 pipeline.py validate
if ($LASTEXITCODE -ne 0) { throw 'Submission validation failed.' }
