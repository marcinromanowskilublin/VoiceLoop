# VoiceLoop: bezpieczny smoke test bez mikrofonu i bez akcji ekranowych.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$base = 'http://127.0.0.1:8765/api/v1'
$tokenPath = Join-Path $projectRoot 'data\voiceloop.token'

if (-not (Test-Path $tokenPath -PathType Leaf)) {
    throw 'Rdzen nie byl uruchomiony. Uzyj scripts\start-core.bat.'
}
$headers = @{ 'X-VoiceLoop-Token' = (Get-Content -Raw -Encoding UTF8 $tokenPath).Trim() }

Write-Host '[1/3] health...'
$health = Invoke-RestMethod "$base/health"
Write-Host "   core=$($health.components.core.status), lm=$($health.components.lm_studio.status), n8n=$($health.components.n8n.status)"

Write-Host '[2/3] polecenie voice_test...'
$body = @{
    schema_version = 1
    source = 'api'
    command_id = 'voice_test'
    include_screen = $false
    allow_cloud = $false
} | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri "$base/commands" -Headers $headers `
    -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
Write-Host "   status=$($result.status), request_id=$($result.request_id)"

Write-Host '[3/3] odczyt stanu...'
Start-Sleep -Milliseconds 250
$command = Invoke-RestMethod "$base/commands/$($result.request_id)" -Headers $headers
Write-Host "   status=$($command.status), intent=$($command.intent)"

if ($command.status -ne 'succeeded') {
    throw "Smoke test zakonczyl sie statusem $($command.status)."
}
Write-Host 'VoiceLoop smoke test: OK'
