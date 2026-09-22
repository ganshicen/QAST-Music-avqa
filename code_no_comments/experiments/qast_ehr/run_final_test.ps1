$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$Manifest = Join-Path $PSScriptRoot "validation_selection_manifest.json"
$Log = Join-Path $PSScriptRoot "final_test.log"
Set-Location $Root
& $Python -u "experiments\qast_ehr\evaluate_final_test.py" --manifest $Manifest --output-dir $PSScriptRoot *>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
