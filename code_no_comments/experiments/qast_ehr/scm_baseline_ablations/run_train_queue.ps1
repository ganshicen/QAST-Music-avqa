$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
$ManifestPath = Join-Path $PSScriptRoot "validation_manifest.json"
$Encoding = [System.Text.UTF8Encoding]::new($false)

$Runs = @(
    [ordered]@{
        Name = "scm_baseline_without_cmg"
        Config = "experiments\qast_ehr\scm_baseline_ablations\configs\without_cmg_seed713.py"
        OutputDir = "D:\model\qast_ehr_scm_baseline_without_cmg_seed713"
    },
    [ordered]@{
        Name = "scm_baseline_without_adaptive_position"
        Config = "experiments\qast_ehr\scm_baseline_ablations\configs\without_adaptive_position_seed713.py"
        OutputDir = "D:\model\qast_ehr_scm_baseline_without_adaptive_position_seed713"
    },
    [ordered]@{
        Name = "scm_baseline_without_patch_grounder"
        Config = "experiments\qast_ehr\scm_baseline_ablations\configs\without_patch_grounder_seed713.py"
        OutputDir = "D:\model\qast_ehr_scm_baseline_without_patch_grounder_seed713"
    }
)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Append-Log([string]$Path, [string]$Line) {
    [IO.File]::AppendAllText([IO.Path]::GetFullPath($Path), $Line + [Environment]::NewLine, $Encoding)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Results = @()
$PreviousLocation = Get-Location
Set-Location $Root
try {
    foreach ($Run in $Runs) {
        $ConfigPath = (Resolve-Path -LiteralPath $Run.Config).Path
        $LogPath = Join-Path $LogDir ($Run.Name + "_train.log")
        $ValidationPath = Join-Path $PSScriptRoot ($Run.Name + "_validation.json")
        $Before = @{}
        if (Test-Path -LiteralPath $Run.OutputDir) {
            Get-ChildItem -LiteralPath $Run.OutputDir -Recurse -Filter best.pt -File | ForEach-Object {
                $Before[$_.FullName] = Get-Sha256 $_.FullName
            }
        }
        $Start = [DateTime]::UtcNow
        Append-Log $LogPath "QUEUE_RUN_START $($Run.Name) $([Guid]::NewGuid().ToString('N'))"
        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python -u "src\train.py" --config $ConfigPath --mode train --seed 713 *>&1 |
            ForEach-Object {
                $Line = [string]$_
                Write-Output $Line
                Append-Log $LogPath $Line
            }
        $ExitCode = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($ExitCode -ne 0) {
            throw "Training failed for $($Run.Name), exit code $ExitCode"
        }
        $CurrentRunLog = Get-Content -LiteralPath $LogPath -Raw
        if (-not ($CurrentRunLog.Contains("training epoch 15") -and $CurrentRunLog.Contains("Epoch 15 done"))) {
            throw "Epoch 15 completion evidence missing for $($Run.Name)"
        }
        $Checkpoint = Get-ChildItem -LiteralPath $Run.OutputDir -Recurse -Filter best.pt -File |
            Where-Object {
                $_.LastWriteTimeUtc -gt $Start -and
                ((-not $Before.ContainsKey($_.FullName)) -or $Before[$_.FullName] -ne (Get-Sha256 $_.FullName))
            } |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -eq $Checkpoint) {
            throw "No new best checkpoint found for $($Run.Name)"
        }
        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python "experiments\qast_ehr\evaluate_validation.py" `
            --config $ConfigPath `
            --checkpoint $Checkpoint.FullName `
            --output $ValidationPath
        $ValidationExitCode = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($ValidationExitCode -ne 0) {
            throw "Validation failed for $($Run.Name)"
        }
        $Result = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json
        $Result | Add-Member -NotePropertyName name -NotePropertyValue $Run.Name -Force
        $Result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ValidationPath -Encoding UTF8
        $Results += $Result
    }
    $Manifest = [ordered]@{
        protocol = "SCM-off baseline; each remaining module removed separately; seed 713; random initialization; 15 epochs"
        generated_at = [DateTime]::UtcNow.ToString("o")
        baseline = [ordered]@{
            name = "without_scm"
            checkpoint = "D:\model\qast_ehr_ablation_without_scm_seed713\2026-09-19-13-43-55_seed713\best.pt"
            test_overall = 77.37977872713331
        }
        results = $Results
    }
    [IO.File]::WriteAllText($ManifestPath, ($Manifest | ConvertTo-Json -Depth 10), $Encoding)
    Write-Host "QUEUE_COMPLETE $ManifestPath"
}
finally {
    Set-Location $PreviousLocation
}
