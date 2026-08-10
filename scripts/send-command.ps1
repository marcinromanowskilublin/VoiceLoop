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

function Get-VoiceLoopHealth {
    try {
        return Invoke-VoiceLoopRequest -Method Get -Path '/health'
    } catch {
        return $null
    }
}

function Resolve-AutoRouteText {
    param([string]$CommandId)

    switch ($CommandId) {
        'voice_test' { return 'test glosu' }
        'open_calendar' { return 'otworz kalendarz' }
        'open_browser' { return 'otworz przegladarke' }
        'search_web' { return 'wyszukaj w internecie' }
        'open_chat' { return 'otworz czat' }
        'open_gpt_chat' { return 'otworz chat gpt' }
        'open_gemini_chat' { return 'otworz gemini' }
        'active_window' { return 'co mam otwarte' }
        'recent_activity' { return 'co robilem ostatnio' }
        'describe_active_window' { return 'co mam otwarte' }
        'describe_recent_activity' { return 'co robilem ostatnio' }
        'describe_text_target' { return 'gdzie teraz pisze' }
        'minimize_active_window' { return 'zminimalizuj okno' }
        'minimize_all_windows' { return 'zminimalizuj wszystkie okna' }
        'copy_selected_text' { return 'kopiuj zaznaczony tekst' }
        'copy_number_under_cursor' { return 'kopiuj numer pod kursorem' }
        'copy_sentence_under_cursor' { return 'kopiuj cale zdanie pod kursorem' }
        default { return '' }
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
            Speak-Text 'Wlaczam nasluch.'
            Invoke-VoiceLoopRequest -Method Post -Path '/listening/start' | Out-Null
            exit 0
        }
        'listen-stop' {
            Invoke-VoiceLoopRequest -Method Post -Path '/listening/stop' | Out-Null
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
            $pending = $commands |
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
                } |
                Select-Object -First 1
            if ($null -eq $pending) {
                Speak-Text 'Nie ma polecenia oczekujacego na decyzje.'
                exit 0
            }
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

    if (-not $Text -and $CommandId -and $Operation -eq 'command') {
        $health = Get-VoiceLoopHealth
        $conversationMode = $false
        if ($health -and $health.components -and $health.components.deepgram) {
            $conversationMode = ($health.components.deepgram.status -eq 'ok')
        }

        $mappedText = Resolve-AutoRouteText -CommandId $CommandId
        if ($conversationMode) {
            if ($mappedText) {
                # W trybie rozmowy zawsze idziemy ścieżką tekstową.
                $Text = $mappedText
                $CommandId = $null
            } else {
                Invoke-VoiceLoopRequest -Method Post -Path '/listening/once?mode=assistant' | Out-Null
                exit 0
            }
        } elseif (-not $mappedText) {
            # Poza trybem rozmowy nieznane polecenie przełączamy na pojedynczy nasłuch.
            Invoke-VoiceLoopRequest -Method Post -Path '/listening/once?mode=assistant' | Out-Null
            exit 0
        }
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
