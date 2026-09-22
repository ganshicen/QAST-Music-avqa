$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
$ManifestPath = Join-Path $PSScriptRoot "validation_manifest.json"
$Utf8 = [System.Text.UTF8Encoding]::new($false)

$Names = @(
    "without_evidence_router",
    "without_event_memory",
    "without_temporal_reader",
    "without_dynamic_fusion",
    "without_tacl",
    "without_sacl",
    "without_event_loss",
    "without_evidence_loss",
    "without_diversity_loss",
    "without_position_loss"
)

function Get-OutputDir([string]$Name) {
    return "D:\model\qast_ehr_remaining_${Name}_seed713"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($Path), $Text, $Utf8)
}

function Append-Utf8([string]$Path, [string]$Text) {
    [IO.File]::AppendAllText([IO.Path]::GetFullPath($Path), $Text + [Environment]::NewLine, $Utf8)
}

function Get-LatestCheckpoint([string]$OutputDir) {
    if (-not (Test-Path -LiteralPath $OutputDir)) { return $null }
    return Get-ChildItem -LiteralPath $OutputDir -Recurse -Filter best.pt -File |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}

function Get-CompletedResult(
    [string]$Name,
    [string]$ConfigPath,
    [string]$LogPath,
    [string]$ValidationPath,
    [string]$OutputDir
) {
    if (-not (Test-Path -LiteralPath $LogPath) -or -not (Test-Path -LiteralPath $ValidationPath)) {
        return $null
    }
    $Log = Get-Content -LiteralPath $LogPath -Raw
    if (-not ($Log.Contains("training epoch 15") -and $Log.Contains("Epoch 15 done"))) {
        return $null
    }
    $Checkpoint = Get-LatestCheckpoint $OutputDir
    if ($null -eq $Checkpoint) { return $null }
    try { $Result = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json } catch { return $null }
    if ([string]$Result.name -cne $Name) { return $null }
    if ([IO.Path]::GetFullPath([string]$Result.config) -ine [IO.Path]::GetFullPath($ConfigPath)) { return $null }
    if ([IO.Path]::GetFullPath([string]$Result.checkpoint) -ine $Checkpoint.FullName) { return $null }
    if ([string]$Result.sha256 -ine (Get-Sha256 $Checkpoint.FullName)) { return $null }
    return $Result
}

function Save-Manifest($Results) {
    $Manifest = [ordered]@{
        protocol = "w/o SCM baseline; remaining structural and auxiliary-loss ablations; seed 713; random initialization; 15 epochs"
        generated_at = [DateTime]::UtcNow.ToString("o")
        order = $Names
        completed = @($Results).Count
        results = @($Results)
    }
    Write-Utf8 $ManifestPath ($Manifest | ConvertTo-Json -Depth 10)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Results = @()
$PreviousLocation = Get-Location
Set-Location $Root
try {
    foreach ($Name in $Names) {
        $ConfigPath = (Resolve-Path "experiments\qast_ehr\remaining_ablations\configs\${Name}_seed713.py").Path
        $OutputDir = Get-OutputDir $Name
        $LogPath = Join-Path $LogDir "${Name}_train.log"
        $ValidationPath = Join-Path $PSScriptRoot "validation_${Name}.json"
        $Completed = Get-CompletedResult $Name $ConfigPath $LogPath $ValidationPath $OutputDir
        if ($null -ne $Completed) {
            Write-Host "SKIP_COMPLETED $Name"
            $Results += $Completed
            Save-Manifest $Results
            continue
        }

        $Before = @{}
        if (Test-Path -LiteralPath $OutputDir) {
            Get-ChildItem -LiteralPath $OutputDir -Recurse -Filter best.pt -File | ForEach-Object {
                $Before[$_.FullName] = Get-Sha256 $_.FullName
            }
        }
        $Start = [DateTime]::UtcNow
        $Marker = "QUEUE_RUN_START $Name $([Guid]::NewGuid().ToString('N'))"
        Append-Utf8 $LogPath $Marker
        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python -u "src\train.py" --config $ConfigPath --mode train --seed 713 *>&1 |
            ForEach-Object {
                $Line = [string]$_
                Write-Output $Line
                Append-Utf8 $LogPath $Line
            }
        $TrainExit = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($TrainExit -ne 0) { throw "Training failed for $Name with exit code $TrainExit" }

        $FullLog = Get-Content -LiteralPath $LogPath -Raw
        $Offset = $FullLog.LastIndexOf($Marker)
        if ($Offset -lt 0) { throw "Missing run marker for $Name" }
        $CurrentLog = $FullLog.Substring($Offset)
        if (-not ($CurrentLog.Contains("training epoch 15") -and $CurrentLog.Contains("Epoch 15 done"))) {
            throw "Epoch 15 evidence missing for $Name"
        }
        $Checkpoint = Get-ChildItem -LiteralPath $OutputDir -Recurse -Filter best.pt -File |
            Where-Object {
                $_.LastWriteTimeUtc -gt $Start -and
                ((-not $Before.ContainsKey($_.FullName)) -or $Before[$_.FullName] -ne (Get-Sha256 $_.FullName))
            } |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -eq $Checkpoint) { throw "No checkpoint created by $Name" }

        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python "experiments\qast_ehr\evaluate_validation.py" --config $ConfigPath --checkpoint $Checkpoint.FullName --output $ValidationPath
        $ValidationExit = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($ValidationExit -ne 0) { throw "Validation failed for $Name" }
        $Result = Get-Content -LiteralPath $ValidationPath -Raw | ConvertFrom-Json
        $Result | Add-Member -NotePropertyName name -NotePropertyValue $Name -Force
        Write-Utf8 $ValidationPath ($Result | ConvertTo-Json -Depth 10)
        $Results += $Result
        Save-Manifest $Results
    }
    Write-Host "QUEUE_COMPLETE $ManifestPath"
}
finally {
    Set-Location $PreviousLocation
}
