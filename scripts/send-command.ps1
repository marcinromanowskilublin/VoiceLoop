param(
    [string]$CommandId,
    [string]$Text,
    [switch]$IncludeScreen,
    [ValidateSet(
        'command',
        'listen-once',
        'listen-start',
        'listen-stop',
        'status',
        'confirm-last',
        'cancel-last'
    )]
    [string]$Operation = 'command',
    [ValidateSet('assistant', 'note', 'remember')]
    [string]$Mode = 'assistant'
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

function Invoke-VoiceLoopRequest {
    param(
        [ValidateSet('Get', 'Post')]
        [string]$Method,
        [string]$Path,
        [object]$Body
    )

    $parameters = @{
        Method  = $Method
        Uri     = "$baseUrl$Path"
        Headers = $headers
    }
    if ($null -ne $Body) {
        $json = $Body | ConvertTo-Json -Depth 8
        $parameters.ContentType = 'application/json; charset=utf-8'
        $parameters.Body = [System.Text.Encoding]::UTF8.GetBytes($json)
    }
    Invoke-RestMethod @parameters
}

function Speak-Text {
    param([string]$Message)

    $clean = $Message.Trim()
    if (-not $clean) {
        return
    }
    try {
        Add-Type -AssemblyName System.Speech
        $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
        try {
            $speaker.SelectVoice('Microsoft Paulina Desktop')
        } catch {
            # Uzyj domyslnego glosu systemowego.
        }
        try {
            $speaker.Speak($clean.Substring(0, [Math]::Min(1000, $clean.Length)))
        } finally {
            $speaker.Dispose()
        }
    } catch {
        Write-Warning "Nie udalo sie odtworzyc komunikatu glosowego: $($_.Exception.Message)"
    }
}

try {
    if ($CommandId -eq 'stop') {
        Invoke-VoiceLoopRequest -Method Post -Path '/stop' | Out-Null
        exit 0
    }

    switch ($Operation) {
        'listen-once' {
            $encodedMode = [uri]::EscapeDataString($Mode)
            Invoke-VoiceLoopRequest -Method Post -Path "/listening/once?mode=$encodedMode" |
                Out-Null
            exit 0
        }
        'listen-start' {
            # Serwer sam mówi przy conversation/start; tu bez lokalnego TTS.
            try {
                Invoke-VoiceLoopRequest -Method Post -Path '/conversation/start' | Out-Null
            } catch {
                Invoke-VoiceLoopRequest -Method Post -Path '/listening/start' | Out-Null
                Speak-Text 'Wlaczam nasluch.'
            }
            exit 0
        }
        'listen-stop' {
            try {
                Invoke-VoiceLoopRequest -Method Post -Path '/conversation/stop' | Out-Null
            } catch {
                Invoke-VoiceLoopRequest -Method Post -Path '/listening/stop' | Out-Null
            }
            Speak-Text 'Nasluch wylaczony.'
            exit 0
        }
        'status' {
            $health = Invoke-VoiceLoopRequest -Method Get -Path '/health'
            $model = if ($health.components.cloud_llm.status -eq 'ok') {
                'Venice'
            } elseif ($health.components.lm_studio.status -eq 'ok') {
                'lokalny Qwen'
            } else {
                'niedostepny'
            }
            $listening = if ($health.components.deepgram.status -eq 'ok') {
                'Nasluch Deepgram jest wlaczony.'
            } else {
                'Nasluch Deepgram jest wylaczony.'
            }
            if ($health.status -eq 'ok') {
                Speak-Text "VoiceLoop dziala. Model: $model. $listening"
            } else {
                Speak-Text (
                    "VoiceLoop dziala w trybie ograniczonym. Model: $model. " +
                    "$listening Sprawdz panel stanu."
                )
            }
            Write-Output ($health | ConvertTo-Json -Depth 8 -Compress)
            exit 0
        }
        { $_ -in 'confirm-last', 'cancel-last' } {
            # Windows PowerShell 5.1 does not enumerate a top-level JSON array
            # when Invoke-RestMethod is used directly inside @(...).
            $commandResponse = Invoke-VoiceLoopRequest -Method Get -Path '/commands?limit=50'
            $commands = @($commandResponse)
            $cutoff = [DateTimeOffset]::UtcNow.AddMinutes(-5)
            $pendingList = @(
                $commands |
                    Where-Object {
                        if ($_.status -ne 'awaiting_confirmation') {
                            return $false
                        }
                        if (
                            $Operation -eq 'confirm-last' -and
                            ($null -eq $_.plan -or @($_.plan.steps).Count -eq 0)
                        ) {
                            return $false
                        }
                        try {
                            return [DateTimeOffset]::Parse([string]$_.updated_at) -ge $cutoff
                        } catch {
                            return $false
                        }
                    }
            )
            if ($pendingList.Count -eq 0) {
                Speak-Text 'Nie ma polecenia oczekujacego na decyzje.'
                exit 0
            }
            if ($pendingList.Count -gt 1) {
                Speak-Text 'Jest kilka oczekujacych potwierdzen. Potwierdz konkretne zadanie z panelu.'
                exit 0
            }
            $pending = $pendingList[0]
            $decision = if ($Operation -eq 'confirm-last') { 'confirm' } else { 'cancel' }
            $requestId = [uri]::EscapeDataString([string]$pending.request_id)
            $decisionResult = Invoke-VoiceLoopRequest -Method Post `
                -Path "/commands/$requestId/$decision"
            if ($Operation -eq 'confirm-last') {
                if ($decisionResult.status -eq 'cancelled') {
                    Speak-Text 'Potwierdzenie wygaslo. Powtorz polecenie.'
                } else {
                    Speak-Text 'Potwierdzono. Wykonuje.'
                }
            } else {
                Speak-Text 'Zadanie anulowane.'
            }
            exit 0
        }
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
        # Znane frazy sa rozstrzygane deterministycznie przed uzyciem LLM.
        allow_cloud = -not [string]::IsNullOrWhiteSpace($Text)
    }
    $response = Invoke-VoiceLoopRequest -Method Post -Path '/commands' -Body $body
    Write-Output ($response | ConvertTo-Json -Depth 8 -Compress)
} catch {
    Speak-Text 'VoiceLoop nie wykonal polecenia. Sprawdz, czy rdzen jest uruchomiony.'
    throw
}
