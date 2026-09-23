$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = "C:\Users\xiaogan\anaconda3\envs\qastr2\python.exe"
$LogDir = Join-Path $PSScriptRoot "logs"
$ManifestPath = Join-Path $PSScriptRoot "validation_manifest.json"
$Utf8 = [System.Text.UTF8Encoding]::new($false)
$Seeds = @(123, 456)

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

function Get-Completed([int]$Seed, [string]$ConfigPath, [string]$OutputDir, [string]$LogPath, [string]$ValidationPath) {
    if (-not (Test-Path $LogPath) -or -not (Test-Path $ValidationPath)) { return $null }
    $Text = Get-Content $LogPath -Raw
    if (-not ($Text.Contains("training epoch 15") -and $Text.Contains("Epoch 15 done"))) { return $null }
    $Checkpoint = Get-LatestCheckpoint $OutputDir
    if ($null -eq $Checkpoint) { return $null }
    try { $Result = Get-Content $ValidationPath -Raw | ConvertFrom-Json } catch { return $null }
    if ([int]$Result.seed -ne $Seed) { return $null }
    if ([string]$Result.sha256 -ine (Get-Sha256 $Checkpoint.FullName)) { return $null }
    if ([IO.Path]::GetFullPath([string]$Result.checkpoint) -ine $Checkpoint.FullName) { return $null }
    if ([IO.Path]::GetFullPath([string]$Result.config) -ine [IO.Path]::GetFullPath($ConfigPath)) { return $null }
    return $Result
}

function Save-Manifest($Results) {
    $Manifest = [ordered]@{
        protocol = "QAST-EHR; seeds 713, 123, 456; random initialization; 15 epochs; best validation checkpoint"
        generated_at = [DateTime]::UtcNow.ToString("o")
        seeds = @(713, 123, 456)
        results = @($Results)
    }
    Write-Utf8 $ManifestPath ($Manifest | ConvertTo-Json -Depth 10)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Existing713Path = Join-Path $Root "experiments\qast_ehr\ablations\validation_without_scm.json"
$Existing713 = Get-Content $Existing713Path -Raw | ConvertFrom-Json
$Existing713 | Add-Member -NotePropertyName seed -NotePropertyValue 713 -Force
$Results = @($Existing713)
$PreviousLocation = Get-Location
Set-Location $Root
try {
    foreach ($Seed in $Seeds) {
        $ConfigPath = (Resolve-Path "experiments\qast_ehr\without_scm_multiseed\configs\without_scm_seed${Seed}.py").Path
        $OutputDir = "D:\model\qast_ehr_ablation_without_scm_seed${Seed}"
        $LogPath = Join-Path $LogDir "seed${Seed}_train.log"
        $ValidationPath = Join-Path $PSScriptRoot "validation_seed${Seed}.json"
        $Completed = Get-Completed $Seed $ConfigPath $OutputDir $LogPath $ValidationPath
        if ($null -ne $Completed) {
            Write-Host "SKIP_COMPLETED seed $Seed"
            $Results += $Completed
            Save-Manifest $Results
            continue
        }

        $Before = @{}
        if (Test-Path $OutputDir) {
            Get-ChildItem $OutputDir -Recurse -Filter best.pt -File | ForEach-Object {
                $Before[$_.FullName] = Get-Sha256 $_.FullName
            }
        }
        $Start = [DateTime]::UtcNow
        $Marker = "QUEUE_RUN_START seed${Seed} $([Guid]::NewGuid().ToString('N'))"
        Append-Utf8 $LogPath $Marker
        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python -u "src\train.py" --config $ConfigPath --mode train --seed $Seed *>&1 |
            ForEach-Object {
                $Line = [string]$_
                Write-Output $Line
                Append-Utf8 $LogPath $Line
            }
        $TrainExit = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($TrainExit -ne 0) { throw "Training failed for seed $Seed" }

        $CurrentLog = (Get-Content $LogPath -Raw)
        $Offset = $CurrentLog.LastIndexOf($Marker)
        if ($Offset -lt 0) { throw "Run marker missing for seed $Seed" }
        $CurrentLog = $CurrentLog.Substring($Offset)
        if (-not ($CurrentLog.Contains("training epoch 15") -and $CurrentLog.Contains("Epoch 15 done"))) {
            throw "Epoch 15 evidence missing for seed $Seed"
        }
        $Checkpoint = Get-ChildItem $OutputDir -Recurse -Filter best.pt -File |
            Where-Object {
                $_.LastWriteTimeUtc -gt $Start -and
                ((-not $Before.ContainsKey($_.FullName)) -or $Before[$_.FullName] -ne (Get-Sha256 $_.FullName))
            } |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -eq $Checkpoint) { throw "No new best checkpoint for seed $Seed" }

        $OldPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python "experiments\qast_ehr\evaluate_validation.py" --config $ConfigPath --checkpoint $Checkpoint.FullName --output $ValidationPath
        $ValidationExit = $LASTEXITCODE
        $ErrorActionPreference = $OldPreference
        if ($ValidationExit -ne 0) { throw "Validation failed for seed $Seed" }
        $Result = Get-Content $ValidationPath -Raw | ConvertFrom-Json
        $Result | Add-Member -NotePropertyName seed -NotePropertyValue $Seed -Force
        Write-Utf8 $ValidationPath ($Result | ConvertTo-Json -Depth 10)
        $Results += $Result
        Save-Manifest $Results
    }
    Write-Host "QUEUE_COMPLETE $ManifestPath"
}
finally {
    Set-Location $PreviousLocation
}
