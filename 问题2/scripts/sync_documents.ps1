param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [string]$Round5Root,
    [string]$Destination = (Join-Path $PSScriptRoot '..\docs')
)
# Publish verified documents without changing their bytes or replacing teammate edits.
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$target = [IO.Path]::GetFullPath($Destination)
$manifest = [Collections.Generic.List[object]]::new()
function Copy-Verified([string]$From, [string]$To, [string]$SourceLabel) {
    $hash = (Get-FileHash -LiteralPath $From -Algorithm SHA256).Hash.ToLowerInvariant()
    if (Test-Path -LiteralPath $To) {
        if ((Get-FileHash -LiteralPath $To -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {
            throw "Destination differs; review before replacing: $To"
        }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $To) | Out-Null
        Copy-Item -LiteralPath $From -Destination $To
    }
    if ((Get-FileHash -LiteralPath $To -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash) {
        throw "Copy verification failed: $To"
    }
    $manifest.Add([ordered]@{
        path = [IO.Path]::GetRelativePath($target, $To).Replace('\','/')
        source = $SourceLabel; sha256 = $hash; bytes = (Get-Item -LiteralPath $To).Length
    })
}
function Copy-Tree([string]$RelativeSource, [string]$RelativeTarget, [string[]]$Extensions) {
    $folder = Join-Path $source $RelativeSource
    foreach ($file in Get-ChildItem -LiteralPath $folder -Recurse -File) {
        if ($file.Extension -notin $Extensions -or $file.Name -like '*_preview.png') { continue }
        $suffix = [IO.Path]::GetRelativePath($folder, $file.FullName)
        Copy-Verified $file.FullName (Join-Path $target (Join-Path $RelativeTarget $suffix)) ([IO.Path]::GetRelativePath($source, $file.FullName).Replace('\','/'))
    }
}
Copy-Tree 'reports' 'round1/reports' @('.docx','.md','.png')
Copy-Tree 'round2/reports' 'round2/reports' @('.docx','.md','.png')
foreach ($round in @('round3','round4')) {
    Copy-Tree "$round/reports" "$round/reports" @('.docx','.md','.json','.tex')
    Copy-Tree "$round/figures" "$round/figures" @('.png','.svg','.pdf','.md','.json')
}
foreach ($relative in @(
    'round3/audit/document_qa.json', 'round3/audit/release_acceptance.json',
    'round4/audit/release_verification.json', 'round4/study/frozen.json',
    'round4/study/protocol.json', 'round4/study/confirmed.json',
    'round4/analysis/class_tradeoff_protocol.json', 'round4/analysis/bootstrap_protocol.json'
)) {
    $dest = $relative.Replace('/audit/','/evidence/').Replace('/study/','/evidence/').Replace('/analysis/','/evidence/')
    Copy-Verified (Join-Path $source $relative) (Join-Path $target $dest) $relative
}
if ($Round5Root) {
    foreach ($relative in @('study/protocol.json','study/development.json','audit/control_reproduction.json','runs/defer2_fold0/epochs.json')) {
        Copy-Verified (Join-Path $Round5Root $relative) (Join-Path $target "round5_snapshot/$relative") "round5/$relative"
    }
}
if (Test-Path -LiteralPath (Join-Path $source 'round5/reports')) {
    Copy-Tree 'round5/reports' 'round5/reports' @('.docx','.md','.json','.tex')
    Copy-Tree 'round5/figures' 'round5/figures' @('.png','.svg','.pdf','.json')
    foreach ($relative in @(
        'audit/poststudy_checks.json','audit/interruption_closure.json','audit/document_qa.json',
        'recovery_evidence/recovery_verification_complete.json',
        'recovery_evidence/serialization_roundtrips.json','recovery_evidence/tests.log',
        'analysis/internal_folds.csv','analysis/decision_table.csv','study/protocol.json'
    )) {
        $folder = if ($relative.StartsWith('analysis/') -or $relative.StartsWith('study/')) {
            $relative
        } else { 'evidence/'+[IO.Path]::GetFileName($relative) }
        Copy-Verified (Join-Path $source "round5/$relative") (Join-Path $target "round5/$folder") "round5/$relative"
    }
}
if (Test-Path -LiteralPath (Join-Path $source 'round6/reports')) {
    Copy-Tree 'round6/reports' 'round6/reports' @('.md')
    Copy-Tree 'round6/study' 'round6/study' @('.json')
    Copy-Tree 'round6/audit' 'round6/audit' @('.json')
}

# A byte-identical copy retains the prior rendered-document QA, not a new QA claim.
$r3qa = Get-Content -LiteralPath (Join-Path $target 'round3/evidence/document_qa.json') -Raw | ConvertFrom-Json
$r4qa = Get-Content -LiteralPath (Join-Path $target 'round4/evidence/release_verification.json') -Raw | ConvertFrom-Json
foreach ($entry in @(@('round3',$r3qa), @('round4',$r4qa))) {
    foreach ($extension in @('docx','md')) {
        $file = @(Get-ChildItem -LiteralPath (Join-Path $target "$($entry[0])/reports") -Filter "*.$extension")
        if ($file.Count -ne 1) { throw 'Expected exactly one main document per round and format' }
        $key = if ($extension -eq 'docx') { 'docx_sha256' } else { 'markdown_sha256' }
        if ((Get-FileHash -LiteralPath $file[0].FullName).Hash.ToLowerInvariant() -ne $entry[1].$key) {
            throw "Previously reviewed document hash mismatch: $($file[0].FullName)"
        }
    }
}
if (Test-Path -LiteralPath (Join-Path $target 'round5/evidence/document_qa.json')) {
    $qa5 = Get-Content -LiteralPath (Join-Path $target 'round5/evidence/document_qa.json') -Raw | ConvertFrom-Json
    foreach ($extension in @('docx','md')) {
        $file = @(Get-ChildItem -LiteralPath (Join-Path $target 'round5/reports') -Filter "*.$extension")
        $key = if ($extension -eq 'docx') { 'docx_sha256' } else { 'markdown_sha256' }
        if ($file.Count -ne 1 -or (Get-FileHash -LiteralPath $file[0].FullName).Hash.ToLowerInvariant() -ne $qa5.$key) {
            throw 'Round-five reviewed document hash mismatch'
        }
    }
}
$confirmed = Get-Content -LiteralPath (Join-Path $target 'round4/evidence/confirmed.json') -Raw | ConvertFrom-Json
if (!$confirmed.stable_improvement -or $confirmed.records.Count -ne 3) { throw 'Missing confirmed three-seed result' }
function Mean([object[]]$Values) { ($Values | Measure-Object -Average).Average }
$rows = foreach ($name in @('accuracy','macro_f1','mae','neutral_f1','positive_f1','missing_f1','missing_mae','score')) {
    $values = @{}
    foreach ($model in @('reference','metrics')) {
        $v = @($confirmed.records | ForEach-Object {
            $m = $_.$model
            switch ($name) {
                'neutral_f1' { $m.clean.class_f1[1] }
                'positive_f1' { $m.clean.class_f1[2] }
                'missing_f1' { $m.selection.mean_missing_macro_f1 }
                'missing_mae' { $m.selection.mean_missing_mae }
                'score' { $m.selection.score }
                default { $m.clean.$name }
            }
        })
        $mean = Mean $v
        $ss = ($v | ForEach-Object { [Math]::Pow($_-$mean,2) } | Measure-Object -Sum).Sum
        $values[$model] = @{mean=$mean; sample_std=[Math]::Sqrt($ss/($v.Count-1)); seeds=$v}
    }
    [ordered]@{metric=$name; round3=$values.reference; round4=$values.metrics; delta=$values.metrics.mean-$values.reference.mean}
}
$summary = [ordered]@{
    generated_utc = [DateTime]::UtcNow.ToString('o')
    source = 'round4/evidence/confirmed.json'; split='official_valid'; n=728
    seeds=@($confirmed.records | ForEach-Object { $_.seed })
    uncertainty='sample standard deviation across three fixed seeds; not a confidence interval'
    metrics=@($rows); paired_confirmation=$confirmed.pairs
}
$summaryPath = Join-Path $target 'iteration_metrics.json'
[IO.File]::WriteAllText($summaryPath, ($summary | ConvertTo-Json -Depth 12)+"`n", [Text.UTF8Encoding]::new($false))
$manifestPath = Join-Path $target 'sync_manifest.json'
[IO.File]::WriteAllText($manifestPath, ([ordered]@{
    synchronized_utc=[DateTime]::UtcNow.ToString('o'); files=@($manifest.ToArray())
    prior_document_qa_hashes_verified=@('round3','round4','round5')
    note='Documents and figures copied without edits. Historical first/second rounds are not the current model. No raw PKL data, weights, or full prediction archives included.'
} | ConvertTo-Json -Depth 6)+"`n", [Text.UTF8Encoding]::new($false))
Write-Output "Verified $($manifest.Count) copied files; recomputed three-seed comparison."
