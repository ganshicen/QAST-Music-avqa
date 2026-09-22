$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
$ManifestPath = Join-Path $PSScriptRoot "ablation_validation_manifest.json"

$Order = @(
    "without_cmg",
    "without_scm",
    "without_adaptive_position",
    "without_patch_grounder"
)

$Runs = @{
    without_cmg = [ordered]@{
        Config = "experiments\qast_ehr\ablations\configs\without_cmg_seed713.py"
        OutputDir = "D:\model\qast_ehr_ablation_without_cmg_seed713"
        TrainLog = Join-Path $LogDir "without_cmg_train.log"
        Validation = Join-Path $PSScriptRoot "validation_without_cmg.json"
    }
    without_scm = [ordered]@{
        Config = "experiments\qast_ehr\ablations\configs\without_scm_seed713.py"
        OutputDir = "D:\model\qast_ehr_ablation_without_scm_seed713"
        TrainLog = Join-Path $LogDir "without_scm_train.log"
        Validation = Join-Path $PSScriptRoot "validation_without_scm.json"
    }
    without_adaptive_position = [ordered]@{
        Config = "experiments\qast_ehr\ablations\configs\without_adaptive_position_seed713.py"
        OutputDir = "D:\model\qast_ehr_ablation_without_adaptive_position_seed713"
        TrainLog = Join-Path $LogDir "without_adaptive_position_train.log"
        Validation = Join-Path $PSScriptRoot "validation_without_adaptive_position.json"
    }
    without_patch_grounder = [ordered]@{
        Config = "experiments\qast_ehr\ablations\configs\without_patch_grounder_seed713.py"
        OutputDir = "D:\model\qast_ehr_ablation_without_patch_grounder_seed713"
        TrainLog = Join-Path $LogDir "without_patch_grounder_train.log"
        Validation = Join-Path $PSScriptRoot "validation_without_patch_grounder.json"
    }
}

function Get-LatestCheckpoint {
    param([Parameter(Mandatory = $true)][string]$OutputDir)

    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $OutputDir -Filter "best.pt" -File -Recurse |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

function Get-CheckpointSnapshot {
    param([Parameter(Mandatory = $true)][string]$OutputDir)

    $Snapshot = @{}
    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        return $Snapshot
    }
    foreach ($Checkpoint in Get-ChildItem -LiteralPath $OutputDir -Filter "best.pt" -File -Recurse) {
        $Snapshot[$Checkpoint.FullName] = [pscustomobject]@{
            LastWriteTimeUtc = $Checkpoint.LastWriteTimeUtc
            Sha256 = Get-CheckpointSha256 $Checkpoint.FullName
        }
    }
    return $Snapshot
}

function Get-LatestRunCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [Parameter(Mandatory = $true)][hashtable]$PreviousSnapshot,
        [Parameter(Mandatory = $true)][DateTime]$TrainingStartedAtUtc
    )

    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        return $null
    }

    $Eligible = @()
    foreach ($Checkpoint in Get-ChildItem -LiteralPath $OutputDir -Filter "best.pt" -File -Recurse) {
        if ($Checkpoint.LastWriteTimeUtc -le $TrainingStartedAtUtc) {
            continue
        }
        $Previous = $PreviousSnapshot[$Checkpoint.FullName]
        if ($null -eq $Previous) {
            $Eligible += $Checkpoint
            continue
        }
        $CurrentHash = Get-CheckpointSha256 $Checkpoint.FullName
        if ($CurrentHash -ine [string]$Previous.Sha256) {
            $Eligible += $Checkpoint
        }
    }
    return $Eligible |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

function Test-Epoch15Evidence {
    param([Parameter(Mandatory = $true)][string]$Text)

    return (
        $Text.Contains("training epoch 15") -and
        $Text.Contains("Epoch 15 done")
    )
}

function Get-CheckpointSha256 {
    param([Parameter(Mandatory = $true)][string]$Checkpoint)

    $Stream = [IO.File]::OpenRead([IO.Path]::GetFullPath($Checkpoint))
    try {
        $Hasher = [Security.Cryptography.SHA256]::Create()
        try {
            $Bytes = $Hasher.ComputeHash($Stream)
        }
        finally {
            $Hasher.Dispose()
        }
    }
    finally {
        $Stream.Dispose()
    }
    return -join ($Bytes | ForEach-Object { $_.ToString("x2") })
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $Encoding = [System.Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($Path), $Content, $Encoding)
}

function Append-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $Encoding = [System.Text.UTF8Encoding]::new($false)
    [IO.File]::AppendAllText([IO.Path]::GetFullPath($Path), $Content, $Encoding)
}

function Test-PathMatch {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right
    )

    try {
        return [IO.Path]::GetFullPath($Left) -ieq [IO.Path]::GetFullPath($Right)
    }
    catch {
        return $false
    }
}

function Get-ResumableResult {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$TrainLog,
        [Parameter(Mandatory = $true)][string]$ValidationPath,
        $Checkpoint
    )

    if ($null -eq $Checkpoint) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $TrainLog -PathType Leaf)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $ValidationPath -PathType Leaf)) {
        return $null
    }
    if (-not (Test-Epoch15Evidence (Get-Content -LiteralPath $TrainLog -Raw))) {
        return $null
    }

    try {
        $Result = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json
        $ActualHash = Get-CheckpointSha256 $Checkpoint.FullName
    }
    catch {
        return $null
    }

    if ([string]$Result.name -cne $Name) {
        return $null
    }
    if (-not (Test-PathMatch ([string]$Result.config) $ConfigPath)) {
        return $null
    }
    if (-not (Test-PathMatch ([string]$Result.checkpoint) $Checkpoint.FullName)) {
        return $null
    }
    if ([string]$Result.sha256 -ine $ActualHash) {
        return $null
    }
    return $Result
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Results = @()
$SeenOutputDirs = @{}

$PreviousLocation = Get-Location
Set-Location $Root
try {
    foreach ($Name in $Order) {
        $Run = $Runs[$Name]
        $ConfigPath = (Resolve-Path -LiteralPath $Run.Config).Path

        & $Python "experiments\qast_ehr\ablations\audit_ablations.py" --config $ConfigPath
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation audit failed for $Name."
        }

        $OutputDir = [string]$Run.OutputDir
        $OutputKey = [IO.Path]::GetFullPath($OutputDir).ToLowerInvariant()
        if ($SeenOutputDirs.ContainsKey($OutputKey)) {
            throw "Ablation output_dir is not unique: $OutputDir."
        }
        $SeenOutputDirs[$OutputKey] = $Name

        $Checkpoint = Get-LatestCheckpoint $OutputDir
        $Completed = Get-ResumableResult `
            -Name $Name `
            -ConfigPath $ConfigPath `
            -TrainLog $Run.TrainLog `
            -ValidationPath $Run.Validation `
            -Checkpoint $Checkpoint
        if ($null -ne $Completed) {
            Write-Host "Skipping completed ablation $Name."
            $Results += $Completed
            continue
        }

        $CheckpointSnapshot = Get-CheckpointSnapshot $OutputDir
        $RunMarker = "QUEUE_RUN_START $Name $([Guid]::NewGuid().ToString('N'))"
        Append-Utf8NoBom -Path $Run.TrainLog -Content ($RunMarker + [Environment]::NewLine)
        $TrainingStartedAtUtc = [DateTime]::UtcNow
        & $Python -u "src\train.py" --config $ConfigPath --mode train --seed 713 *>&1 |
            ForEach-Object {
                $Line = [string]$_
                Write-Output $Line
                Append-Utf8NoBom -Path $Run.TrainLog -Content ($Line + [Environment]::NewLine)
            }
        $TrainExitCode = $LASTEXITCODE
        if ($TrainExitCode -ne 0) {
            throw "Training failed for $Name with exit code $TrainExitCode."
        }

        $FullLog = Get-Content -LiteralPath $Run.TrainLog -Raw
        $MarkerOffset = $FullLog.LastIndexOf($RunMarker)
        if ($MarkerOffset -lt 0) {
            throw "Training marker was not preserved in $($Run.TrainLog)."
        }
        $CurrentRunLog = $FullLog.Substring($MarkerOffset)
        if (-not (Test-Epoch15Evidence $CurrentRunLog)) {
            throw "Training log for $Name does not prove epoch 15 completed."
        }

        $Checkpoint = Get-LatestRunCheckpoint `
            -OutputDir $OutputDir `
            -PreviousSnapshot $CheckpointSnapshot `
            -TrainingStartedAtUtc $TrainingStartedAtUtc
        if ($null -eq $Checkpoint) {
            throw "No best.pt created or changed by the current $Name training run."
        }

        & $Python "experiments\qast_ehr\evaluate_validation.py" `
            --config $ConfigPath `
            --checkpoint $Checkpoint.FullName `
            --output $Run.Validation
        if ($LASTEXITCODE -ne 0) {
            throw "Validation failed for $Name."
        }

        $Result = Get-Content -LiteralPath $Run.Validation -Raw | ConvertFrom-Json
        $Result | Add-Member -NotePropertyName "name" -NotePropertyValue $Name -Force
        $ValidationJson = $Result | ConvertTo-Json -Depth 10
        Write-Utf8NoBom -Path $Run.Validation -Content $ValidationJson

        $ExpectedHash = Get-CheckpointSha256 $Checkpoint.FullName
        if ([string]$Result.sha256 -ine $ExpectedHash) {
            throw "Validation SHA256 does not match checkpoint for $Name."
        }
        if (-not (Test-PathMatch ([string]$Result.checkpoint) $Checkpoint.FullName)) {
            throw "Validation checkpoint does not match latest best.pt for $Name."
        }
        if (-not (Test-PathMatch ([string]$Result.config) $ConfigPath)) {
            throw "Validation config does not match $Name."
        }
        $Results += $Result
    }

    $Manifest = [ordered]@{
        protocol = "four predetermined 15-epoch ablations trained sequentially and evaluated on validation only"
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        order = $Order
        results = $Results
    }
    $ManifestJson = $Manifest | ConvertTo-Json -Depth 10
    Write-Utf8NoBom -Path $ManifestPath -Content $ManifestJson
    Write-Host "QAST-EHR ablation validation queue complete: $ManifestPath"
}
finally {
    Set-Location $PreviousLocation
}
