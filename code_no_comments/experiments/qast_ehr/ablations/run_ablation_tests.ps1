$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$Manifest = Join-Path $PSScriptRoot "ablation_validation_manifest.json"
$OutputDir = $PSScriptRoot
$ConsumedFlag = Join-Path $OutputDir "ablation_tests_consumed.flag"
$Log = Join-Path $OutputDir "ablation_test.log"

if (Test-Path -LiteralPath $ConsumedFlag) {
    Write-Error "Ablation tests have already been consumed: $ConsumedFlag"
    exit 2
}

$PreviousLocation = Get-Location
$ExitCode = 1
Set-Location $Root
try {
    & $Python -u "experiments\qast_ehr\ablations\evaluate_ablation_tests.py" `
        --manifest $Manifest `
        --output-dir $OutputDir *>&1 |
        Tee-Object -FilePath $Log
    $ExitCode = $LASTEXITCODE
}
finally {
    Set-Location $PreviousLocation
}
exit $ExitCode
