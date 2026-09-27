param(
    [Parameter(Mandatory=$true)][string]$SourcePath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    Write-Host 'Opening saved Word cover...'
    $document = $word.Documents.Open([ref]$SourcePath, [ref]$false, [ref]$true, [ref]$false)
    # Clear footer text only in the temporary in-memory document.
    foreach ($section in $document.Sections) {
        foreach ($footer in $section.Footers) {
            if ($footer.Exists) { $footer.Range.Text = '' }
        }
    }
    $document.Repaginate()
    Write-Host 'Exporting first page...'
    # PDF format 17; range 3 means explicit page range, here page 1 only.
    $document.ExportAsFixedFormat($OutputPath, 17, $false, 0, 3, 1, 1)
    Write-Host 'First-page PDF exported.'
} catch {
    Write-Host ('Word conversion error: ' + $_.Exception.Message)
    throw
} finally {
    if ($null -ne $document) {
        $document.Close([ref]0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit([ref]0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
