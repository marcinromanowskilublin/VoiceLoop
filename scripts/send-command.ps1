param(
    [string]$CommandId,
    [string]$Text,
    [switch]$IncludeScreen
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$tokenPath = Join-Path $projectRoot 'data\voiceloop.token'
$baseUrl = 'http://127.0.0.1:8765/api/v1'

if (-not (Test-Path $tokenPath -PathType Leaf)) {
    throw "Brak tokenu VoiceLoop: $tokenPath. Najpierw uruchom scripts\start-core.bat."
}
$token = (Get-Content -Raw -Encoding UTF8 $tokenPath).Trim()
$headers = @{ 'X-VoiceLoop-Token' = $token }

if ($CommandId -eq 'stop') {
    Invoke-RestMethod -Method Post -Uri "$baseUrl/stop" -Headers $headers | Out-Null
    exit 0
}

if (-not $CommandId -and -not $Text) {
    throw 'Podaj -CommandId albo -Text.'
}

$body = @{
    schema_version = 1
    source = 'voiceattack'
    command_id = if ($CommandId) { $CommandId } else { $null }
    text = if ($Text) { $Text } else { $null }
    include_screen = [bool]$IncludeScreen
    allow_cloud = $false
} | ConvertTo-Json
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)

$response = Invoke-RestMethod -Method Post -Uri "$baseUrl/commands" `
    -Headers $headers -ContentType 'application/json; charset=utf-8' -Body $bodyBytes

Write-Output ($response | ConvertTo-Json -Depth 8 -Compress)
