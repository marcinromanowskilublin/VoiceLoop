# VoiceLoop — hybrydowy asystent Windows

VoiceLoop łączy polski głos, Venice AI, lokalne modele w LM Studio, Screenpipe,
n8n, UI.Vision RPA i VoiceAttack w jedną kontrolowaną pętlę.

Szczegółowa dokumentacja architektury, handoff dla kolejnego AI i wersja PDF:

- [`docs/VOICELOOP_ARCHITECTURE_HANDOFF.md`](docs/VOICELOOP_ARCHITECTURE_HANDOFF.md)
- [`docs/VOICELOOP_ARCHITECTURE_HANDOFF.pdf`](docs/VOICELOOP_ARCHITECTURE_HANDOFF.pdf)

```text
mikrofon / panel / VoiceAttack
             │
             ▼
     rdzeń VoiceLoop :8765
       ├─ Deepgram Nova-3 (pl)
       ├─ Venice AI (główny planer)
       ├─ LM Studio Qwen (fallback)
       ├─ LM Studio Nomic (embeddingi)
       ├─ Screenpipe + Qdrant (5 named vectors)
       ├─ SQLite (stan, historia i dual-write)
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
- Venice `venice-uncensored-1-2` jako główny planer,
- Qwen `qwen2.5-14b-instruct-1m-abliterated` jako lokalny fallback,
- Nomic `text-embedding-nomic-embed-text-v2-moe` jako model embeddingowy,
- ustrukturyzowane planowanie i wieloetapowy format planu,
- lokalne metadane aktywnego okna i opcjonalny zrzut dla planera,
- pamięć SQLite z jawnym potwierdzeniem zapisu,
- lokalny Qdrant z wektorami `semantic`, `topic`, `intent`, `decision`
  i `person_context`,
- ciągłe lokalne trawienie aktywności przez Qwen także podczas pracy użytkownika,
- szybkie wyszukiwanie internetowe (`search_web`) przez DuckDuckGo lub Brave,
- Deepgram Nova-3 dla polskiego mikrofonu,
- n8n Router v1 na `http://127.0.0.1:5678`,
- UI.Vision z wynikiem, logiem i timeoutem,
- lokalny polski TTS Microsoft Paulina,
- przycisk STOP i deduplikacja komend.

VoiceAttack wymaga jednorazowego ręcznego importu profilu, opisanego niżej.

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

1. załaduj `qwen2.5-14b-instruct-1m-abliterated` jako lokalny fallback,
2. załaduj `text-embedding-nomic-embed-text-v2-moe` do embeddingów,
3. otwórz **Developer → Local Server**,
4. uruchom serwer na porcie `1234`.

Konfiguracja została ograniczona do `127.0.0.1`, CORS jest wyłączony, a
logowanie treści wrażliwych wyłączone. Jeśli LM Studio przywróci ustawienia po
aktualizacji, health check pokaże problem.

### 2. Wszystkie usługi

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-all.ps1
```

Skrypt:

- sprawdza LM Studio,
- synchronizuje makra UI.Vision do runtime,
- uruchamia Qdrant w trwałym kontenerze Docker na `127.0.0.1:6333`,
- uruchamia Screenpipe z pełnym lokalnym zapisem i retencją 14 dni,
- uruchamia n8n, jeśli nie działa,
- uruchamia rdzeń VoiceLoop,
- czeka domyślnie do 300 sekund na wszystkie porty,
- otwiera panel.

Limit można zmienić, np.:

```powershell
.\scripts\start-all.ps1 -TimeoutSeconds 600
```

Można też uruchamiać osobno:

```text
scripts\start-n8n.bat
scripts\start-core.bat
powershell -ExecutionPolicy Bypass -File .\scripts\start-qdrant.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-screenpipe.ps1
```

Screenpipe zapisuje dwa monitory, mikrofon, fizyczne wyjścia audio oraz zdarzenia
wejścia, w tym klawiaturę, kliknięcia, schowek, przewijanie, okna prywatne oraz
projekcje `memory` i `automation`. Redakcja tekstu, obrazów i PII jest wyłączona.
API nadal nasłuchuje tylko na localhost. Globalny silnik transkrypcji Screenpipe
jest wyłączony. VoiceLoop wysyła do Deepgram wyłącznie audio zakończonych spotkań
wykrytych przez Screenpipe; domeny i okna YouTube są blokowane bez wyjątków.

### 3. Panel

Otwórz:

```text
http://127.0.0.1:8765
```

Panel umożliwia:

- wpisanie polecenia lub uruchomienie mikrofonu,
- opcjonalne dołączenie aktywnego okna do planowania,
- potwierdzanie i anulowanie ryzykownych kroków,
- STOP,
- przegląd zadań oraz lokalnej pamięci.

Panel pobiera aktywny tryb LLM z lokalnej sesji. Przy `LLM_PRIMARY=cloud`
informuje, że Venice jest modelem głównym i wyłącza checkbox fallbacku. W trybie
local-first checkbox jawnie zezwala wyłącznie na chmurowy fallback.

## n8n bez Dockera

Obecnie używana i przetestowana wersja to `n8n 2.33.7`, instalowana przez npm.
Workflow `n8n\voice-loop.json` jest opublikowany jako **VoiceLoop Router v1**.
n8n nasłuchuje wyłącznie na `127.0.0.1`, a Execute Command jest wyłączony.

n8n 2.33.7 wyświetla ostrzeżenie, że uruchamianie npm poza kontenerem będzie
wycofywane w przyszłych wersjach. Nie wpływa to na bieżącą wersję, ale przed
aktualizacją n8n trzeba sprawdzić migrację do oficjalnego obrazu.

Aktualny workflow nie weryfikuje nagłówka `X-VoiceLoop-Token`; jego ochroną
jest obecnie wyłącznie bind do loopback. Portu `5678` nie wolno wystawiać do
sieci bez dodania autoryzacji webhooka.

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

- akceptuje wyłącznie bezpieczną nazwę makra z allowlisty repozytorium,
- wymaga zsynchronizowanej kopii w runtime,
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
2. wybierz `voiceattack\VoiceLoop-v2.vap`
3. ustaw aktywny profil **VoiceLoop v2**

Pełna lista 17 komend, instalacja i test przyjęcia:
[`voiceattack/INSTRUKCJA.md`](voiceattack/INSTRUKCJA.md).

Skrypty `.vbs` są skierowane do jednego lokalnego dispatchera
`scripts\send-command.ps1`. VoiceAttack nie powinien być uruchamiany jako
administrator, jeśli sterowana aplikacja nie wymaga podniesionych uprawnień.

Najważniejsza komenda działa w dwóch krokach:

1. powiedz **„Asystent”**;
2. po komunikacie **„Słucham”** wypowiedz dowolne polskie polecenie.

VoiceAttack rozpoznaje tylko pewną frazę wybudzającą. Następną wypowiedź
transkrybuje Deepgram, jednorazowy nasłuch sam się zamyka, a VoiceLoop kieruje
tekst przez n8n/router do Venice (primary) lub Qwen (fallback). Osobne stałe
komendy obsługują notatkę, pamięć z potwierdzeniem, aktywne okno, aktywność
Screenpipe, stan usług, nasłuch oraz natychmiastowy STOP.

STOP omija n8n i bezpośrednio anuluje rdzeń.

## Konfiguracja

Wzór znajduje się w `listener\.env.example`. Najważniejsze pola:

```text
DEEPGRAM_API_KEY=
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=pl
SAMPLE_RATE=16000
AUTO_START_LISTENING=false
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=qwen2.5-14b-instruct-1m-abliterated
N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/voice-command-v1
N8N_TIMEOUT_SECONDS=5
LLM_PRIMARY=cloud
CLOUD_LLM_ENABLED=true
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_FALLBACK_PROVIDER=duckduckgo
WEB_SEARCH_GEMINI_MODEL=gemini-3.6-flash
```

`LLM_PRIMARY=local` używa LM Studio jako głównego mózgu. `LLM_PRIMARY=cloud`
lub `LLM_PRIMARY=venice` używa providera chmurowego jako głównego mózgu,
a LM Studio zostaje fallbackiem.

Provider chmurowy musi udostępniać API zgodne z OpenAI:

```text
LLM_PRIMARY=cloud
CLOUD_LLM_ENABLED=true
CLOUD_LLM_BASE_URL=https://api.venice.ai/api/v1
CLOUD_LLM_API_KEY=
CLOUD_LLM_MODEL=venice-uncensored-1-2
```

Aktualna instalacja używa Venice jako głównego modelu. Klucz API pozostaje
wyłącznie w lokalnym `listener\.env`.

Główny planer otrzymuje historię, pamięć i opcjonalny obraz. Planer fallback
otrzymuje ograniczony kontekst bez historii, pamięci oraz obrazu.

### Lokalna pamięć Qdrant ze Screenpipe

VoiceLoop stale buduje pamięć z aktywności i transkrypcji Screenpipe:

```text
Screenpipe → Qwen: analiza i wnioski
           → Nomic: 5 embeddingów
           → Qdrant: zapis i wyszukiwanie
           → SQLite: dual-write i stan operacyjny
```

Każdy nowy punkt ma pięć named vectors: `semantic`, `topic`, `intent`,
`decision` i `person_context`. Model Qwen działa cyklicznie także podczas
aktywnej pracy; nie ma warunku bezczynności.

Najważniejsze ustawienia:

```text
LOCAL_EMBEDDINGS_ENABLED=true
LOCAL_EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v2-moe
VECTOR_MEMORY_CONTEXT_LIMIT=8
QDRANT_ENABLED=true
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=voiceloop_memory
QDRANT_DUAL_WRITE=true
BEHAVIOR_DIGEST_ENABLED=true
BEHAVIOR_DIGEST_MODEL=qwen2.5-14b-instruct-1m-abliterated
BEHAVIOR_DIGEST_TIMEOUT_SECONDS=240
BEHAVIOR_DIGEST_POLL_SECONDS=60
BEHAVIOR_DIGEST_RECENT_MINUTES=30
SCREENPIPE_VECTOR_MEMORY_ENABLED=true
SCREENPIPE_VECTOR_POLL_SECONDS=120
SCREENPIPE_VECTOR_RECENT_MINUTES=60
```

Surowe dane Screenpipe i pełne wektory zostają lokalnie. Planer dostaje tylko
kilka najbardziej trafnych fragmentów z Qdrant, po połączeniu wyników pięciu
przestrzeni cosine. Istniejące wektory SQLite są migrowane bez usuwania starej
bazy.

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
potwierdzenia, kolejkę, routing modeli, embeddingi i worker Screenpipe.

## Struktura

```text
VoiceLoop\
├── docs\
│   ├── VOICELOOP_ARCHITECTURE_HANDOFF.md
│   └── VOICELOOP_ARCHITECTURE_HANDOFF.pdf
├── listener\
│   ├── voiceloop\          rdzeń FastAPI
│   ├── requirements.in     zależności bez pinu
│   ├── requirements.lock   dokładny stan środowiska
│   └── start-listener.bat
├── panel\index.html        lokalny panel
├── n8n\voice-loop.json     bezpieczny router
├── scripts\
│   ├── start-all.ps1
│   ├── start-qdrant.ps1
│   ├── start-screenpipe.ps1
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
- Obraz trafia do Venice tylko wtedy, gdy request ma `include_screen=true`.
- SSE i webhook n8n opierają ochronę na lokalnym bindzie; nie wystawiaj portów.
- Stary adres `panel\deepgram.html` przekierowuje do aktualnego panelu.
