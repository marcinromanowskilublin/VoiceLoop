[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$UiVisionHome = (Join-Path $env:USERPROFILE 'Desktop\uivision')
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot 'uivision'
$runtimeMacros = Join-Path $UiVisionHome 'macros'
$runtimeImages = Join-Path $UiVisionHome 'images'

New-Item -ItemType Directory -Force -Path $runtimeMacros | Out-Null

foreach ($macro in Get-ChildItem (Join-Path $sourceRoot 'macros') -Filter '*.json' -File) {
    Get-Content -Raw -Encoding UTF8 $macro.FullName | ConvertFrom-Json | Out-Null
    $destination = Join-Path $runtimeMacros $macro.Name
    if ($PSCmdlet.ShouldProcess($destination, "Copy $($macro.Name)")) {
        Copy-Item -Force $macro.FullName $destination
    }
}

$sourceImages = Join-Path $sourceRoot 'images'
if (Test-Path $sourceImages -PathType Container) {
    New-Item -ItemType Directory -Force -Path $runtimeImages | Out-Null
    foreach ($image in Get-ChildItem $sourceImages -File) {
        $destination = Join-Path $runtimeImages $image.Name
        if ($PSCmdlet.ShouldProcess($destination, "Copy $($image.Name)")) {
            Copy-Item -Force $image.FullName $destination
        }
    }
}

Write-Output "UI.Vision zsynchronizowany: $UiVisionHome"
