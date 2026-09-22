$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Results = @()
Set-Location $Root

foreach ($Name in @("a", "b", "c")) {
    $Config = Join-Path $PSScriptRoot "configs\qast_ehr_${Name}_seed713.py"
    & $Python "experiments\qast_ehr\audit_protocol.py" --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Audit failed for candidate $Name" }
    $TrainLog = Join-Path $LogDir "candidate_${Name}_train.log"
    & $Python -u "src\train.py" --config $Config --mode train --seed 713 *>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) { throw "Training failed for candidate $Name" }
    $OutputDir = & $Python -c "import importlib.util; p=r'$Config'; s=importlib.util.spec_from_file_location('c',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.config['output_dir'])"
    $Checkpoint = Get-ChildItem -LiteralPath $OutputDir -Filter best.pt -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $Checkpoint) { throw "No best.pt found for candidate $Name" }
    $ResultPath = Join-Path $PSScriptRoot "validation_${Name}.json"
    & $Python "experiments\qast_ehr\evaluate_validation.py" --config $Config --checkpoint $Checkpoint.FullName --output $ResultPath
    if ($LASTEXITCODE -ne 0) { throw "Validation failed for candidate $Name" }
    $Results += Get-Content -LiteralPath $ResultPath -Raw | ConvertFrom-Json
}

$Eligible = @($Results | Where-Object { $_.eligible_for_test } | Sort-Object overall -Descending)
if ($Eligible.Count -gt 0) {
    $Selected = $Eligible[0]
} else {
    $Selected = @($Results | Sort-Object overall -Descending)[0]
}
$Manifest = [ordered]@{
    protocol = "three predetermined 15-epoch candidates selected only by validation"
    generated_at = (Get-Date).ToString("o")
    selected = $Selected
    candidates = $Results
}
$ManifestPath = Join-Path $PSScriptRoot "validation_selection_manifest.json"
$Manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
Write-Host "QAST-EHR validation queue complete: $ManifestPath"
if (-not $Selected.eligible_for_test) { exit 2 }
