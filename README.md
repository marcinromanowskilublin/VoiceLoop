# VoiceLoop — hybrydowy asystent Windows

VoiceLoop łączy polski głos, lokalny model w LM Studio, n8n, UI.Vision RPA
i VoiceAttack w jedną kontrolowaną pętlę:

```text
mikrofon / panel / VoiceAttack
             │
             ▼
     rdzeń VoiceLoop :8765
       ├─ Deepgram Nova-3 (pl)
       ├─ LM Studio (plan + vision)
       ├─ pamięć SQLite
       ├─ zgody, kolejka i STOP
       └─ n8n (routing + integracje)
             │
             ▼
 Windows API / UI.Vision / TTS / workflow n8n
```

LLM i n8n zwracają wyłącznie typowane `action_id` oraz argumenty. Nie mogą
przekazać dowolnego polecenia powłoki do wykonania.

## Aktualny stan

Działa pełny pionowy przekrój:

- panel i lokalne API na `http://127.0.0.1:8765`,
- LM Studio OpenAI API na `http://127.0.0.1:1234/v1`,
- model `mistral-small-3.1-24b-instruct-2503`,
- ustrukturyzowane planowanie i wieloetapowy format planu,
- lokalne rozumienie aktywnego okna (zrzut + UI Automation),
- pamięć SQLite z jawnym potwierdzeniem zapisu,
- Deepgram Nova-3 dla polskiego mikrofonu,
- n8n Router v1 na `http://127.0.0.1:5678`,
- UI.Vision z wynikiem, logiem i timeoutem,
- lokalny polski TTS Microsoft Paulina,
- przycisk STOP i deduplikacja komend.

Adapter chmurowy jest gotowy konfiguracyjnie, ale domyślnie wyłączony. VoiceAttack
wymaga jednorazowego ręcznego importu profilu, opisanego niżej.

## Ważne: klucz Deepgram

Poprzedni klucz był zapisany również w kodzie starego panelu, więc należy uznać
go za ujawniony:

1. unieważnij go w Deepgram Console,
2. wygeneruj nowy,
3. wpisz go wyłącznie do `listener\.env` jako `DEEPGRAM_API_KEY=...`.

Panel nie przechowuje już klucza. `.env`, baza, logi i środowisko wirtualne są
wykluczone przez `.gitignore`.

## Uruchamianie

### 1. LM Studio

W LM Studio:

1. załaduj model z obsługą structured output i obrazu,
2. otwórz **Developer → Local Server**,
3. uruchom serwer na porcie `1234`.

Konfiguracja została ograniczona do `127.0.0.1`, CORS jest wyłączony, a
logowanie treści wrażliwych wyłączone. Jeśli LM Studio przywróci ustawienia po
aktualizacji, health check pokaże problem.

### 2. Wszystkie usługi

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-all.ps1
```

Skrypt:

- sprawdza LM Studio,
- uruchamia n8n, jeśli nie działa,
- uruchamia rdzeń VoiceLoop,
- otwiera panel.

Można też uruchamiać osobno:

```text
scripts\start-n8n.bat
scripts\start-core.bat
```

### 3. Panel

Otwórz:

```text
http://127.0.0.1:8765
```

Panel umożliwia:

- wpisanie polecenia lub uruchomienie mikrofonu,
- dołączenie aktywnego okna do lokalnego modelu,
- jawne zezwolenie na chmurę dla pojedynczego zadania,
- potwierdzanie i anulowanie ryzykownych kroków,
- STOP,
- przegląd zadań oraz lokalnej pamięci.

## n8n bez Dockera

Obecnie używana i przetestowana wersja to `n8n 2.33.7`, instalowana przez npm.
Workflow `n8n\voice-loop.json` jest opublikowany jako **VoiceLoop Router v1**.
n8n nasłuchuje wyłącznie na `127.0.0.1`, a Execute Command jest wyłączony.

n8n 2.33.7 wyświetla ostrzeżenie, że uruchamianie npm poza kontenerem będzie
wycofywane w przyszłych wersjach. Nie wpływa to na bieżącą wersję, ale przed
aktualizacją n8n trzeba sprawdzić migrację do oficjalnego obrazu.

Kopia workflow sprzed migracji:

```text
n8n\backups\workflows-before-voiceloop.json
```

## UI.Vision

Repozytorium jest źródłem prawdy dla makr:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync-uivision.ps1
```

Runtime znajduje się w:

```text
%USERPROFILE%\Desktop\uivision
```

`ui.vision.html` został utworzony na podstawie oficjalnego generatora UI.Vision,
a dostęp rozszerzenia do adresów `file://` został włączony.

Test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-uivision.ps1 `
  -Macro voiceloop_notatka.json `
  -Var1 "Test VoiceLoop" `
  -TimeoutSeconds 30
```

Runner:

- akceptuje tylko bezpieczną nazwę istniejącego makra,
- koduje `cmd_var1..3`,
- tworzy osobny log każdego uruchomienia,
- czeka na jednoznaczny rezultat,
- kończy się błędem po timeout.

## VoiceAttack

VoiceAttack 2.1.7 jest zainstalowany w:

```text
C:\Program Files\VoiceAttack\VoiceAttack.exe
```

VoiceAttack jest zarejestrowany (pełna wersja). Import profilu:

1. **More Actions → Import Profile**
2. wybierz `voiceattack\VoiceLoop-profil.vap`

Awaryjnie (gdyby import zawiódł): dodaj komendy ręcznie według
`voiceattack\INSTRUKCJA.md`.

VoiceAttack pozostaje uruchamiany jako administrator. Skrypty `.vbs` są już
skierowane do bezpiecznego `scripts\send-command.ps1`.

Komendy profilu:

- `voice test`,
- `open calendar`,
- `open browser`,
- `open chat`,
- `take a note` / `new note`,
- `stop now` / `abort`.

STOP omija n8n i bezpośrednio anuluje rdzeń.

## Konfiguracja

Wzór znajduje się w `listener\.env.example`. Najważniejsze pola:

```text
DEEPGRAM_API_KEY=
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=
N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/voice-command-v1
CLOUD_LLM_ENABLED=false
```

Puste `LM_STUDIO_MODEL` oznacza użycie pierwszego modelu zwróconego przez
`GET /v1/models`.

Provider chmurowy musi udostępniać API zgodne z OpenAI:

```text
CLOUD_LLM_ENABLED=true
CLOUD_LLM_BASE_URL=
CLOUD_LLM_API_KEY=
CLOUD_LLM_MODEL=
```

Historia, pamięć i obraz nie są przekazywane do providera chmurowego. Chmura
otrzymuje tylko bieżące polecenie i wyłącznie po `allow_cloud=true`.

## Lokalne API

Najważniejsze endpointy:

```text
GET  /api/v1/health
POST /api/v1/commands
GET  /api/v1/commands
POST /api/v1/commands/{id}/confirm
POST /api/v1/commands/{id}/cancel
POST /api/v1/stop
POST /api/v1/listening/start
POST /api/v1/listening/stop
GET  /api/v1/memories
POST /api/v1/memories
GET  /api/v1/events
```

Polecenia modyfikujące wymagają lokalnego nagłówka `X-VoiceLoop-Token`. Token
jest generowany przy pierwszym starcie w `data\voiceloop.token`.

Dokumentacja OpenAPI:

```text
http://127.0.0.1:8765/api/docs
```

## Testy

Smoke test bez mikrofonu i bez otwierania aplikacji:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test-loop.ps1
```

Testy Python:

```powershell
cd listener
.\.venv\Scripts\python -m ruff check voiceloop ..\tests
.\.venv\Scripts\python -m pytest -c pyproject.toml -q
```

Zestaw obejmuje routing, normalizację języka polskiego, pamięć, politykę ryzyka,
potwierdzenia, kolejkę i prywatną eskalację modelu.

## Struktura

```text
VoiceLoop\
├── listener\
│   ├── voiceloop\          rdzeń FastAPI
│   ├── requirements.in     zależności bez pinu
│   ├── requirements.lock   dokładny stan środowiska
│   └── start-listener.bat
├── panel\index.html        lokalny panel
├── n8n\voice-loop.json     bezpieczny router
├── scripts\
│   ├── start-all.ps1
│   ├── send-command.ps1
│   ├── run-uivision.ps1
│   ├── sync-uivision.ps1
│   └── va\*.vbs
├── uivision\macros\
├── voiceattack\
├── tests\
├── data\                   baza, token, zrzuty (lokalne)
└── logs\                   logi wykonania (lokalne)
```

## Zasady bezpieczeństwa

- Usługi nasłuchują tylko na loopback.
- LLM nie wykonuje tekstu jako shell.
- Każda akcja istnieje w lokalnej allowliście.
- Model nie może obniżyć poziomu ryzyka akcji.
- Operacje wysokiego ryzyka zawsze wymagają potwierdzenia.
- Jedna akcja ekranowa działa naraz.
- `request_id` i okno deduplikacji zapobiegają podwójnemu wykonaniu.
- STOP przerywa TTS, Deepgram, kolejkę i bieżący proces UI.Vision.
- Obraz jest najpierw przetwarzany lokalnie; provider chmurowy go nie otrzymuje.
