# VoiceLoop — hybrydowy asystent Windows

VoiceLoop łączy polski głos, lokalny rdzeń FastAPI, typowane akcje Windows oraz
opcjonalne modele i usługi: Gemini, Venice AI, LM Studio, Screenpipe, Qdrant,
n8n, UI.Vision RPA i VoiceAttack.

Szczegółowa dokumentacja architektury, handoff dla kolejnego AI i wersja PDF:

- [`docs/VOICELOOP_ARCHITECTURE_HANDOFF.md`](docs/VOICELOOP_ARCHITECTURE_HANDOFF.md)
- [`docs/VOICELOOP_ARCHITECTURE_HANDOFF.pdf`](docs/VOICELOOP_ARCHITECTURE_HANDOFF.pdf)
- [`docs/HANDOFF_MIKOLAJ.md`](docs/HANDOFF_MIKOLAJ.md) — szybki pakiet
  przekazania projektu i checklista odbioru
- [`docs/PROGRAM_DOKUMENTACJA_PL.md`](docs/PROGRAM_DOKUMENTACJA_PL.md) — skrócona
  dokumentacja programu: budowa, działanie, wdrożenie, rozwój
- [`docs/SAFE_USER_CORPUS.md`](docs/SAFE_USER_CORPUS.md) — lokalny korpus,
  zestaw 120 próbek głosowych, prozodia i ewaluacja Routing V2
- [`docs/PORTFOLIO_PL.md`](docs/PORTFOLIO_PL.md) — krótki, uczciwy opis
  portfolio, granice prywatności i scenariusz demo

```text
mikrofon / panel / VoiceAttack
             │
             ▼
     rdzeń VoiceLoop :8765
       ├─ Deepgram Nova-3 (pl) — zawsze najpierw tekst
      ├─ Gemini — główna rozmowa; zadanie trafia do typowanego planera
       ├─ Azure Speech SDK → Azure REST → Windows TTS
       ├─ LM Studio Qwen (fallback planera + trawienie)
       ├─ LM Studio Nomic (embeddingi)
       ├─ Screenpipe + Qdrant (5 named vectors)
       ├─ SQLite (stan, historia i dual-write)
       ├─ zgody, kolejka i STOP
       └─ n8n (opcjonalny router, domyślnie wyłączony)
             │
             ▼
 Windows API / UI.Vision / VoiceAttack / opcjonalny workflow n8n
```

LLM i n8n zwracają wyłącznie typowane `action_id` oraz argumenty. Nie mogą
przekazać dowolnego polecenia powłoki do wykonania.

## Rozmowa vs zadanie (bez rewolucji)

Prosty flow:

1. **Deepgram** zawsze zbiera tekst (ciągły nasłuch albo jednorazowo z VA).
2. Powiedzenie **„Venice…” zawsze otwiera nową sesję rozmowy MAX IQ**.
3. Kolejne wypowiedzi należą do tej samej sesji, zachowują jej kontekst i nie
   wymagają ponownego słowa „Venice”.
4. Znane, twarde komendy (`command_id` z VoiceAttack) idą od razu do akcji.
5. W sesji Venice nadal rozróżnia:
   - rozmowę → `intent=conversation`, naturalna odpowiedź przez Azure TTS,
   - wyraźne zadanie → `intent=task`, kroki z lokalnej allowlisty.

Sterowanie sesją:

- `Venice, porozmawiajmy o...` → zaczyna nową rozmowę i zeruje kontekst
  poprzedniej sesji,
- następne pytania można mówić bez prefiksu,
- `koniec rozmowy` / `zakończ rozmowę` → zamyka sesję,
- `stop` / `przerwij` → po uzbrojeniu mikrofonu barge-in przerywa generowanie,
  mowę Azure i wykonywane akcje, ale pozostawia sesję otwartą; przycisk/komenda
  VoiceAttack STOP i API działają także w okresie ochronnym.

Kontekst sesji obejmuje ostatnie 12 wiadomości użytkownika i asystenta. Jest
trzymany w pamięci procesu: restart rdzenia zamyka sesję, a powiedzenie
„Venice…” zaczyna ją od czystego kontekstu. Jeśli Deepgram nie usłyszy komendy
przez głośną mowę Azure, można użyć stałej komendy **STOP** w VoiceAttack albo
przycisku STOP w panelu.

Poza aktywną sesją Venice nadal decyduje na podstawie treści: pytanie oznacza
rozmowę, a konkretne czasowniki (`otwórz`, `minimalizuj`, `skopiuj`,
`wyszukaj`, `zapamiętaj`) oznaczają zadanie.

Style odpowiedzi:

- sesja rozpoczęta przez `Venice ...` → MAX IQ,
- `Asystencie ...` → zwięźle,
- bez prefiksu i bez aktywnej sesji → balanced.

### Polecenia wtrącone podczas rozmowy

Aktywnej rozmowy nie trzeba przerywać ani przełączać ręcznie do VoiceAttack.
Deepgram przekazuje całe zdanie do Venice, a Venice rozróżnia pytanie od
lokalnego zadania. Przykład:

```text
skopiuj mi ten mail, gdzie mam kursor, do schowka
```

VoiceLoop wykonuje lokalną akcję `copy_email_under_cursor`, mówi krótko
„Skopiowałem adres e-mail” dopiero po sukcesie i pozostawia sesję rozmowy
aktywną. VoiceAttack nadal odpowiada za szybkie komendy stałe, wybudzanie i
awaryjny STOP; nie wymaga osobnej komendy dla każdej naturalnej parafrazy.

Głos: **Azure Speech SDK** (`pl-PL-ZofiaNeural`) z natywnym zatrzymaniem mowy.
Jeśli SDK zawiedzie, VoiceLoop próbuje dotychczasowego Azure REST, a następnie
lokalnego Windows TTS.
Mózg rozmowy: **Gemini** (`LLM_PRIMARY=gemini`) albo **Venice**/lokalny Qwen.
Gemini do wyszukiwania nadal używa osobnego `WEB_SEARCH_*`. Sterowanie PC:
**Deepgram (STT) → VoiceLoop → VoiceAttack/akcje** (np. zmiana nazwy pod kursorem).

### Kontrakt odpowiedzi modelu i diagnostyka opóźnień

Rozmowa i wykonywanie zadań używają dwóch oddzielnych kontraktów:

- `intent=conversation` → model zwraca zwykły, krótki tekst po polsku; TTS nigdy
  nie dostaje schematu ani opisu JSON,
- zadanie → planer zwraca walidowany JSON Schema z `action_id` i argumentami,
- ucięty lub niepoprawny JSON jest błędem protokołu i może uruchomić kontrolowany
  retry/fallback; nie jest zamieniany na odpowiedź rozmowy,
- odpowiedź rozmowy jest odrzucana i generowana ponownie, gdy API zwróci
  `finish_reason=length`, inny niekońcowy status albo tekst bez końcowej
  interpunkcji.

Dla Gemini ustawiany jest niski poziom reasoning, ale odpowiednio duży budżet
completion. Zapobiega to sytuacji, w której tokeny myślenia zużywają cały limit,
a odpowiedź kończy się na „Here is the JSON requested”.

Po każdym powrocie do słuchania log i zdarzenie SSE `conversation.timing`
zawierają:

```text
tts_ms, cooldown_ms, deepgram_reconnect_ms
```

Stan `listening_once` jest publikowany dopiero po rzeczywistym zestawieniu
WebSocketu Deepgram.

### Pełne TTS, przerwa czasowa i ochrona przed rozmową w tle

Barge-in porównuje interim/final Deepgram z aktualnie czytanym tekstem. Echo
głośnika pasujące do TTS — również krótkie, zniekształcone fragmenty 2–3 słów
i końcówka odebrana tuż po ponownym otwarciu mikrofonu — jest ignorowane, więc
asystentka nie wywołuje już `tts.stop()` własnym głosem. Bezpośredni zwrot
`Asystencie…` omija ochronę końcówki. Po okresie ochronnym dokładne
„stop”/„przerwij” działa już na interim; VoiceAttack i API mogą przerwać TTS od
początku.
Inna wypowiedź czeka na final Deepgram przed przerwaniem; sam niestabilny interim
nie ucina zdania.
Zdarzenie `conversation.tts_completed` podaje, czy odczyt zakończył się normalnie
czy został świadomie przerwany.
Timeout Azure SDK i awaryjnej ścieżki REST jest obliczany z długości tekstu oraz
tempa głosu, dzięki czemu dłuższa odpowiedź nie kończy się po stałych 20 sekundach.

Przerwa czasowa jest obsługiwana deterministycznie, bez wysyłania komendy do LLM:

1. powiedz `Przerwij działanie na 10 minut` (liczba może być cyfrą albo polskim
   słowem z zakresu 1–99, np. `dwie minuty`),
2. asystentka zapyta `Czy na pewno wstrzymać działanie na 10 minut?`,
3. powiedz `potwierdzam` albo `anuluj`,
4. podczas przerwy zwykłe wypowiedzi są ignorowane; `wznów działanie`, `włącz się`
   lub `koniec przerwy` wznawia pracę od razu,
5. po zadanym czasie praca wznawia się automatycznie.

Komenda jest przechwytywana także przy zwykłym ciągłym nasłuchu Deepgram, zanim
transkrypt trafi do planera, więc pytanie o potwierdzenie nie zależy od LLM.

Deepgram live używa `diarize_model=latest` i przekazuje identyfikatory mówców.
Wypowiedź obejmująca co najmniej dwóch mówców jest ignorowana, chyba że zaczyna się
od `Asystencie…`/`Venice…`. Dodatkowo po 30 sekundach bez nowej odpowiedzi kończy
się okno dialogu bez wake worda; później trzeba ponownie zwrócić się do asystentki.
To ogranicza przypadkowe reakcje na rozmowę obok, ale diarization nie jest
biometrią: numer `speaker` nie identyfikuje właściciela i zeruje się przy nowym
połączeniu WebSocket. Pełne rozpoznawanie właściciela wymaga osobnego enrollmentu
głosu.

## Aktualny stan

Zweryfikowany rdzeń i dostępne integracje obejmują:

- panel i lokalne API na `http://127.0.0.1:8765`,
- LM Studio OpenAI API na `http://127.0.0.1:1234/v1`,
- Gemini `gemini-3.6-flash` jako konfigurowalny model rozmowy,
- Venice `venice-uncensored-1-2` jako opcjonalny provider chmurowy,
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
- zamrożony lokalny zestaw 120 klipów audio (30 development / 90 holdout) z
  timestampami, hashami, ręcznymi adnotacjami i analizą prozodii,
- opcjonalny n8n Router v1, domyślnie wyłączony,
- UI.Vision z wynikiem, logiem i timeoutem,
- Azure Speech SDK jako główny głos, Azure REST i Windows TTS jako fallbacki,
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
- pozostawia n8n wyłączone, dopóki nie zostanie uruchomione jawnie,
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
jest wyłączony. Automatyczny worker wysyła do Deepgram audio zakończonych spotkań
wykrytych przez Screenpipe; domeny i okna YouTube są blokowane bez wyjątków.
Jawny tryb **Rozpocznij nagranie** omija detekcję spotkania: wyłącza odpowiedzi
asystenta, zapisuje osobno PCM mikrofonu i WASAPI loopback aktywnego outputu,
importuje zapasowe pliki Screenpipe, archiwizuje wszystko w
`data/meetings/<session_id>/audio` i pokazuje diarization Deepgram w panelu.

### 3. Panel

Otwórz:

```text
http://127.0.0.1:8765
```

Panel umożliwia:

- wpisanie polecenia lub uruchomienie mikrofonu,
- rozpoczęcie i zakończenie trwałego nagrania spotkania,
- niezależne włączenie lub wyłączenie live Deepgram bez zatrzymywania Screenpipe,
- podgląd transkryptu z etykietami `Ty` / `Rozmówca`,
- opcjonalne dołączenie aktywnego okna do planowania,
- potwierdzanie i anulowanie ryzykownych kroków,
- STOP,
- przegląd zadań oraz lokalnej pamięci.

Panel pobiera aktywny tryb LLM z lokalnej sesji. Przy `LLM_PRIMARY=cloud`
informuje, że Venice jest modelem głównym i wyłącza checkbox fallbacku. W trybie
local-first checkbox jawnie zezwala wyłącznie na chmurowy fallback.

## n8n bez Dockera — opcjonalnie

Repo zawiera workflow **VoiceLoop Router v1** dla `n8n 2.33.7`. Integracja jest
domyślnie wyłączona (`N8N_ENABLED=false`) i nie jest wymagana przez demo
portfolio. Po jawnym uruchomieniu n8n nasłuchuje wyłącznie na `127.0.0.1`, a
Execute Command pozostaje wyłączone.

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

## Hume — eksperyment, nie funkcja demo

Repo zawiera eksperymentalny klient prozodii Hume oraz testy parsera odpowiedzi.
`HUME_EMOTION_ANALYSIS_ENABLED=false` jest ustawieniem domyślnym. Integracja nie
jest przedstawiana jako zweryfikowana end-to-end ani używana w demo portfolio.
Jej jawne włączenie wysyła fragmenty audio spotkania do chmurowego Hume EVI.

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
3. ustaw aktywny profil **VoiceLoop v2 PRO**

Profil **VoiceLoop v2 PRO** zachowuje stabilny identyfikator profilu v2, ale
zawiera 30 bezpiecznych komend i 622 jawne warianty języka naturalnego
(odmiany, krótkie formy, typowe warianty bez polskich znaków). Pełna lista,
instalacja i test przyjęcia:
[`voiceattack/INSTRUKCJA.md`](voiceattack/INSTRUKCJA.md).

Skrypty `.vbs` są skierowane do jednego lokalnego dispatchera
`scripts\send-command.ps1`. VoiceAttack nie powinien być uruchamiany jako
administrator, jeśli sterowana aplikacja nie wymaga podniesionych uprawnień.

Najważniejsza komenda działa w dwóch krokach:

1. powiedz **„Asystent”**;
2. po komunikacie **„Słucham”** wypowiedz dowolne polskie polecenie.

VoiceAttack rozpoznaje rozbudowany pakiet pewnych fraz. Gdy Deepgram jest
aktywny, dispatcher mapuje znaną komendę VA z powrotem na naturalny tekst, aby
sesja Venice pozostała spójna. Gdy Deepgram jest wyłączony, ta sama fraza leci
bezpośrednio po `command_id`. Pakiet „pod kursorem” obejmuje kopiowanie
dostępnego tekstu, adresu e-mail, numeru i zdania oraz zaznaczanie zdania lub
akapitu przez UI Automation. „Wyłącz aplikację pod kursorem” wymaga
potwierdzenia i wysyła `WM_CLOSE` zamiast zabijać proces, więc aplikacja może
nadal zapytać o zapis zmian. Kursor musi pozostać nad tym samym oknem do chwili
powiedzenia „Potwierdź”. Jeśli program nie udostępnia zakresów tekstowych UI
Automation, zaznaczanie kończy się czytelnym błędem bez awaryjnego klikania.
Dla wypowiedzi spoza profilu użyj `Asystent` albo ciągłego nasłuchu Deepgram.

STOP omija n8n i bezpośrednio anuluje rdzeń.

## Konfiguracja

Wzór znajduje się w `listener\.env.example`. Najważniejsze pola:

```text
DEEPGRAM_API_KEY=
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=pl
SAMPLE_RATE=16000
AUTO_START_LISTENING=false
AUTO_START_CONVERSATION=false
CONVERSATION_GREETING=Cześć. Możemy porozmawiać albo możesz od razu wydać polecenie. Możesz też zapytać, co potrafię. W czym mogę pomóc?
STT_MIN_ACTION_CONFIDENCE=0.75
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=qwen2.5-14b-instruct-1m-abliterated
N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/voice-command-v1
N8N_TIMEOUT_SECONDS=5
LLM_PRIMARY=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.6-flash
CLOUD_LLM_ENABLED=false
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_FALLBACK_PROVIDER=duckduckgo
WEB_SEARCH_GEMINI_MODEL=gemini-3.6-flash
AZURE_TTS_ENABLED=true
AZURE_TTS_KEY=
AZURE_TTS_REGION=germanywestcentral
AZURE_TTS_VOICE=pl-PL-ZofiaNeural
TTS_RATE_PERCENT=-20
TTS_PITCH_PERCENT=-5
```

`azure-cognitiveservices-speech` jest instalowane z `requirements.lock`.
Biblioteka SDK jest bezpłatna; rozliczane pozostaje użycie chmurowej usługi
Azure Speech według wybranego planu, tak samo jak przy REST. SDK odtwarza głos
bez pliku tymczasowego i obsługuje natywne `stop_speaking_async`. Kolejność
awaryjna to: Speech SDK → Azure REST → Windows TTS.

`LLM_PRIMARY=local` używa LM Studio jako głównego mózgu. `LLM_PRIMARY=gemini`
używa skonfigurowanego `GEMINI_MODEL` (`gemini-3.6-flash`) z lokalnym Qwen jako
fallbackiem błędów transportu lub
protokołu.
`LLM_PRIMARY=cloud` lub `LLM_PRIMARY=venice` używa providera chmurowego jako
głównego mózgu, a LM Studio zostaje fallbackiem.

Provider chmurowy musi udostępniać API zgodne z OpenAI:

```text
LLM_PRIMARY=cloud
CLOUD_LLM_ENABLED=true
CLOUD_LLM_BASE_URL=https://api.venice.ai/api/v1
CLOUD_LLM_API_KEY=
CLOUD_LLM_MODEL=venice-uncensored-1-2
```

Wzór konfiguracji pozostawia `LLM_PRIMARY=local`; dostawcę chmurowego włącza
się jawnie. Klucze API pozostają wyłącznie w lokalnym `listener\.env`.

Główny planer otrzymuje historię, pamięć i opcjonalny obraz. Planer fallback
otrzymuje ograniczony kontekst bez historii, pamięci oraz obrazu.

### Lokalna pamięć Qdrant ze Screenpipe

VoiceLoop stale buduje pamięć z aktywności i transkrypcji Screenpipe:

```text
Screenpipe → Qwen: analiza i wnioski
           → filtr jakości i deduplikacja
           → Nomic: 3 albo 5 embeddingów
           → Qdrant: zapis i wyszukiwanie
           → SQLite: dual-write i stan operacyjny
```

Aktywność komputera używa profilu 3-vector: `semantic`, `intent`,
`person_context`. Rozmowy i wizyty używają pełnego profilu 5-vector:
`semantic`, `topic`, `intent`, `decision`, `person_context`. Surowy OCR nie jest
już zapisywany jako normalna pamięć, jeśli lokalny digest ma niską pewność
(`confidence < 0.65`). Typowe elementy interfejsu są czyszczone przed analizą,
a powiązane wcześniejsze obserwacje trafiają do metadanych zamiast doklejać się
do nowej treści. Model Qwen działa cyklicznie także podczas aktywnej pracy; nie
ma warunku bezczynności.

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
BEHAVIOR_DIGEST_POLL_SECONDS=300
BEHAVIOR_DIGEST_RECENT_MINUTES=10
SCREENPIPE_VECTOR_MEMORY_ENABLED=true
SCREENPIPE_VECTOR_POLL_SECONDS=300
SCREENPIPE_VECTOR_RECENT_MINUTES=10
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
POST /api/v1/conversation/start
POST /api/v1/conversation/interrupt
POST /api/v1/conversation/resume
POST /api/v1/conversation/stop
POST /api/v1/listening/start
POST /api/v1/listening/stop
GET  /api/v1/memories
POST /api/v1/memories
GET  /api/v1/events
```

Endpointy prywatne, w tym szczegółowy health i strumień SSE, wymagają lokalnego
nagłówka `X-VoiceLoop-Token`. Token jest generowany przy pierwszym starcie w
`data\voiceloop.token`; panel pobiera go wyłącznie przez endpoint dostępny z
loopback.

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
.\.venv\Scripts\python -m ruff check voiceloop ..\tests `
  ..\scripts\voice_capture_server.py `
  ..\scripts\holding-commands\server.py `
  ..\scripts\calibration-phrases\server.py
.\.venv\Scripts\python -m pytest -c pyproject.toml -q
```

Zestaw obejmuje routing, normalizację języka polskiego, pamięć, politykę ryzyka,
potwierdzenia, kolejkę, routing modeli, embeddingi, worker Screenpipe oraz profil
VoiceAttack. Pakiet zawiera ponad 500 testów; dokładny wynik i pełny Ruff są
weryfikowane przez workflow CI. Generator profilu dodatkowo odrzuca kolizje
fraz i brakujące wrappery VBS.

## Struktura

```text
VoiceLoop\
├── docs\
│   ├── PORTFOLIO_PL.md
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
│   ├── calibration-phrases\
│   ├── holding-commands\
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
- Głosowe `stop` / `przerwij` anuluje model, TTS, kolejkę i bieżącą akcję,
  pozostawiając nasłuch i sesję rozmowy aktywne.
- Globalny STOP w panelu dodatkowo zatrzymuje Deepgram.
- Obraz trafia do Venice tylko wtedy, gdy request ma `include_screen=true`.
- Health, SSE i operacje prywatne wymagają lokalnego tokenu.
- Webhook n8n nadal opiera ochronę na lokalnym bindzie; nie wystawiaj portu.
- Stary adres `panel\deepgram.html` przekierowuje do aktualnego panelu.
