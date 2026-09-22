# VoiceLoop — aktualny baseline architektury

**Wersja:** 0.3.0
**Baseline.** Stan kodu z 22 września 2026, po `bf26abe`.
**Źródło prawdy:** `listener/voiceloop/`. Starsze opisy ustępują kodowi.  
**Ten plik nie zmienia zachowania produkcyjnej pętli.** Planer nadal nie
wykonuje i nie czyta pamięci, timeline'u ani stanu jako źródła intencji.

Opis dotyczy kodu i domyślnych ustawień, nie bieżącego stanu lokalnych usług.
Prywatny `.env` może zmienić flagi i nazwy kolekcji. Wyniki testów nie
potwierdzają działania mikrofonu, modelu ani jakości prywatnych wspomnień.

Rdzeń zostaje:

```text
planer → walidacja → allowlista → polityka → executor
```

Jawny model sytuacji i Context Timeline istnieją jako osobne kontrakty.
Żaden z nich nie jest źródłem akcji dla executora.

---

## Trzy półki — nie mieszać

### DZIAŁA (produkcyjna pętla)

- Wejście → `CommandRequest` → zapis SQLite + dedupe → STOP/pauza → tanie reguły → (opcjonalnie) n8n → plan LLM albo plan V1 → `enforce_policy` → jedna kolejka executora → `ActionResult`.
- LLM zwraca `ProposedPlan`. Lokalny kod wiąże go do `CommandPlan`. Model nie wykonuje akcji i nie dostaje powłoki.
- Allowlista: 44 `ActionSpec` w `listener/voiceloop/actions.py`. 0 akcji `high`.
  9 `medium`, z czego 8 wymaga potwierdzenia.
- `windows_shell.py` = UIA pulpitu i Eksploratora. To nie jest spawn `cmd` / PowerShell dla LLM.
- Pamięć planera i domyślny recall łączą role SQL i Qdrant opisane jako A/B/C
  poniżej. Osobna ścieżka Context Timeline korzysta z FTS5/BM25 i czasu.
- Capabilities żyją w osobnej kolekcji Qdrant; domyślna nazwa to
  `voiceloop_capabilities_v1`, konfigurowana przez `QDRANT_CAPABILITY_COLLECTION`.
- Threshold Guard mierzy progi. Nie zapisuje punktów i nie zastępuje retrievalu.
- `TranscriptEnvelopeV1` jest kontraktem STT. Routing V2 jest w kodzie, ale **nie steruje** produkcją przy domyślnych flagach.

### EKSPERYMENT (kod jest, nie steruje produkcją albo nie jest wpięty)

- Routing V2: `routing_v2_enabled=true`, `shadow_mode=true`, `routing_v2_execute=false`. Liczy się obok V1. Live wymaga quality gate + zgodnego fingerprintu + (opcjonalnie) canary.
- Commitment Layer: detector + scoring + schema. `assistant.py` uruchamia go
  jako `commitment.shadow`. Opcjonalny `CONTEXT_TIMELINE_COMMITMENT_REVIEW_ENABLED`
  zapisuje także wiersz do review w SQLite, z `review_required=true` i
  `executable=false`. Nie jest to akceptacja zobowiązania ani wejście executora.
- `EvidenceItem` w `commitments/schema.py` (`rule` / `vector` / `temporal` / `resolver`) — to **obecny** typ commitmentów. Detector emituje wyłącznie `kind="rule"`. To **nie** jest przyszły `EvidenceItemV1` sytuacji.
- n8n: `n8n_enabled=false`. Webhook dokładnych fraz, bez Execute Command. Pad n8n nie wali asystenta.
- Hume EVI: szkielet, domyślnie off.
- `hybrid` w `voice_conversation.py` = barge-in / reuse strumienia Deepgram, **nie** retrieval.
- Shadow Qdrant (`qdrant_memory_next_collection`) i kalibracja V2 (`routing_v2_calibration_mode=off`) — pomiar, nie sterowanie.
- `EvidenceItemV1` (`situation/evidence.py`) — kontrakt dowodu; nie jest commitment `EvidenceItem` i nie wiąże akcji.
- `SituationStateV1` — ledger in-memory + `GET /api/v1/situation`. Planer nie czyta go jako intencji. LLM nie pisze. Brak tabeli SQL `situation`.
- `StateProposal` (`situation/proposal.py`) — model może proponować (`propose_fact` + `evidence_refs`). Lokalne `StatePolicy` + `StateReducer` decydują i wołają `append_event(actor=local_code)`. **Nie** wpięte w planer / `assistant.py`. Tylko testy / przyszły shadow.
- Context Timeline V1: lokalne tabele zdarzeń/epizodów, FTS5, jawne adaptery
  Screenpipe, spotkań, dokumentów i projektów, migracja pamięci SQL, sampler
  foreground, encje z review gate, selektywne wektory, TimeFirstRetriever,
  review zobowiązań i porównanie shadow. Recall oraz nowe flagi projekcji,
  ekranu deiktycznego i zapisu review są domyślnie wyłączone. Brak
  automatycznych pętli. Kontrakt: [`CONTEXT_TIMELINE_V1.md`](CONTEXT_TIMELINE_V1.md).

### PLANOWANE (nie wpinąć w planer w tym etapie)

- Wpięcie commitment analyzer w trwały produkcyjny stan; obecny shadow event
  pozostaje obserwacyjny.
- Wpięcie `SituationStateV1` w routing / executor jako źródło akcji — **zakazane**.
- Etapy wektorowe i temporalne commitmentów (schema ma sloty, kod ich nie wypełnia).
- Live Routing V2 po quality gate z liczbami.
- Automatyczny prune wektorów (`VECTOR_MEMORY_PRUNE_ENABLED=false`).
- Background ingest/foreground/prune Context Timeline oraz produkcyjny quality
  gate na prywatnym gold secie. Adaptery dokumentów, projektów, migracji
  pamięci i porównania shadow są jawne. Dokumenty, migracja i porównanie mają
  komendy CLI; projekcję projektów udostępnia API i osobna flaga usługi Windows.
- Większy zestaw wyspecjalizowanych osi pamięci oraz jawne reguły nawigacji,
  budżetu i zatrzymania zależne od klasy pytania. Liczba/nazwy osi i kryteria
  wystarczających dowodów pozostają do uzgodnienia:
  [`VECTOR_NAVIGATION_DESIGN.md`](VECTOR_NAVIGATION_DESIGN.md).

Kontrakty dowodu i stanu są w repo i **nie sterują** planerem:

- invariants: [`ARCHITECTURE_INVARIANTS.md`](ARCHITECTURE_INVARIANTS.md)
- dowód: [`EVIDENCE_V1.md`](EVIDENCE_V1.md) — `EvidenceItemV1` ≠ commitment `EvidenceItem`
- stan: [`SITUATION_STATE_V1.md`](SITUATION_STATE_V1.md) — ledger + `GET /api/v1/situation` (read-only)

Nie proponować: Pinecone, chmurowych embeddingów, multi-agent, shella dla LLM, zapisu stanu z LLM, wpięcia stanu w planer.

---

## Diagram: co naprawdę jedzie

```text
                    DZIAŁA                          EKSPERYMENT              PLANOWANE
  panel / Deepgram / VA / API
            │
            ▼
     CommandRequest
     (+ opcjonalnie TranscriptEnvelopeV1)
            │
            ▼
     AssistantService.handle
     zapis SQLite + dedupe 2s
            │
     STOP? ──────────────► interrupt → executor.stop_all → TTS / UI.Vision
     pauza potwierdzona? ► ignoruj transkrypt (poza wznowieniem)
            │
            ▼
     Router V1 (router.py)          Routing V2 shadow          SituationStateV1
     + compound_fast_path_guard     (liczy, nie steruje)       (GET, nie steruje)
            │                            │
     n8n (off) ─ fallback ───────────────┘
            │
            ▼
     kontekst planera
     A: Qdrant 5 osi ⊕ SQL memories (concat)
     B: SQLite vector_memories gdy Qdrant pusty/pad
            │
            ▼
     ProposedPlan (LLM) ──► lokalne wiązanie ──► CommandPlan
            │
            ▼
     ActionRegistry.enforce_policy
            │
            ▼
     CommandExecutor
     kolejka 10 / jeden plan / TTL potwierdzenia 300s
            │
            ▼
     ActionSpec.handler → ActionResult
     Windows / UIA / UI.Vision / TTS / remember|recall

  Commitment Layer ──► shadow event / opcjonalny wiersz review, poza wykonaniem
  Capabilities Qdrant ──► V2 / „co potrafisz”, nie merge z pamięcią
```

---

## Właściciele stanu

Każda warstwa ma jednego właściciela zapisu. Inne warstwy czytają, nie przepisują reguł.

### INPUT — `CommandRequest` + `TranscriptEnvelopeV1`

**Pliki:** `listener/voiceloop/models.py`, `app.py`, `deepgram.py`, `voice_conversation.py`.

Właściciel wejścia: FastAPI składa `CommandRequest` (`schema_version`, `request_id`, `source`, `text` albo `command_id`, `include_screen`, `allow_cloud`, opcjonalnie `transcript`).

Źródła (`CommandSource`): `panel`, `deepgram`, `voiceattack`, `api`, `n8n`.

Głos: Deepgram → `TranscriptEnvelopeV1` (surowy + znormalizowany tekst, pewność, słowa, `speaker_ids`, `is_final` / `speech_final`). `CommandRequest.from_transcript()` kopiuje envelope; tekst i pewność muszą być zgodne z envelope.

`VoiceConversationCoordinator` jest właścicielem sesji głosowej: half-duplex, echo TTS, multi-speaker, pauza z potwierdzeniem (max 24 h), soft barge-in. `POST /api/v1/stop` = soft barge-in (tnij TTS, wróć do słuchania). Twardy koniec sesji: `POST /api/v1/conversation/stop` → `hard_stop()`. Panic / intent `stop` w asystencie woła `assistant.interrupt()` → `executor.stop_all()`.

Słowo `hybrid` tutaj oznacza `conversation_hybrid_barge_in_enabled` + reuse strumienia. To nie jest hybryda retrievalu.

### ROUTING — kto składa kandydata planu, zanim LLM albo zamiast LLM

**Pliki:** `router.py` (V1), `routing/service.py`, `routing/segmenter.py`, `routing/resolver.py`, `routing/assembler.py`, `n8n_client.py`, `assistant.py`.

Kolejność w `AssistantService._handle_request` / `_create_plan`:

1. Intent `stop` — natychmiast, przed lockiem tury.
2. Dedupe fingerprintu tekstu (`COMMAND_DEDUPE_SECONDS=2`).
3. Koniec rozmowy (lokalne frazy) — plan bez executora akcji.
4. `deterministic_plan()` (V1) + bramki STT / multi-speaker.
5. Routing V2: w rozmowie + shadow i bez jawnego czasownika akcji — w tle (nie steruje). Inaczej `evaluate()` synchronicznie. Live tylko gdy `execution_enabled` i `plan_execution_allowed`.
6. Jeśli V1 / V2-live / bramka dały plan chroniony (`stop`, `voice_test`, `list_capabilities`, `stt_confidence_gate`, `speaker_gate`, `compound_fast_path_guard`, albo jawna akcja) — **nie** wołaj LLM.
7. n8n, tylko gdy nie trwa managed conversation; błąd = idź dalej.
8. Retrieval + `ModelRouter.plan()`.

V1 bez modelu: test pętli, capabilities, otwarcia, okna, pulpit, notatka, Screenpipe, STOP, proste remember/recall i reszta reguł w `router.py`. Złożone / niejednoznaczne segmentacje dostają `compound_fast_path_guard` (clarification, bez częściowego wykonania).

V2: segmenter → `CapabilityIndex.search_subtasks(..., min_score=-1.0)` (prefilter wyłączony) → resolver (wagi `vector=0.60`, `lexical=0.35`, `arguments=0.05`; `STT_MIN_ACTION_CONFIDENCE=0.75`; `ROUTING_V2_EXECUTE_MIN_SCORE=0.50`; `ROUTING_V2_EXECUTE_MIN_MARGIN=0.10`; jeden kandydat bez komparatora = clarify) → `assemble_plan()` → `CommandPlan(provider="routing_v2")`. Canary (gdyby live): okna, recall, URL, folder, app — tylko `low` bez potwierdzenia.

Handoff §2 z 11.08 ustawia n8n jako pierwszą produkcyjną ścieżkę i milczy o V2. **Nieaktualne.** Notion z 30.08 jest bliżej kodu: V2 shadow, n8n off.

### MEMORY — ścieżki A/B/C oraz osobna oś czasu z FTS5

**Pliki:** `memory.py`, `qdrant_memory.py`, `memory_vectorization.py`, `manual_memory.py`, `assistant.py` (`_vector_memories_for_request`, `_create_plan`), `actions.py` (`remember` / `recall`), `capability_index.py`.

Dwa zastosowania kolekcji Qdrant, z konfigurowalnymi nazwami. Wersje lub
kolekcje shadow mogą współistnieć; tabela pokazuje ustawienia domyślne:

| Kolekcja | Osie | Po co |
|---|---|---|
| `voiceloop_memory` | `semantic` 0.40, `topic` 0.20, `intent` 0.15, `decision` 0.15, `person_context` 0.10 | kontekst planera i recall wektorowy |
| `voiceloop_capabilities_v1` | `semantic`, `intent`, `target_context` | katalog „co potrafisz” / V2 |

RRF pamięci: `k=60`, `VECTOR_MEMORY_MIN_SCORE=0.0`; ranking łączy pozycje
wyników z poszczególnych osi, nie uśrednia wektorów. Jawne markery pytania
mogą podwoić wagę pasującej osi przed normalizacją. Puste aspekty dokumentu
są pomijane. Schema dokumentów: `memory-documents-v2`.

Klient OpenAI-compatible wysyła osobne teksty z prefiksami `search_document:`
i `search_query:`. Model jest konfigurowalny lub rozpoznawany z `/models`;
wymiar pochodzi z odpowiedzi embeddingu i musi pasować do kolekcji. Nomic
768D jest możliwą konfiguracją lokalną, nie stałym kontraktem całego projektu.

SQL (`data/voiceloop.db`, WAL) ma następujące role:

**A — Context Pack do planera.** `AssistantService._create_plan` zbiera
równolegle Qdrant/fallback B, jawne pamięci, ostatnie akcje i komunikaty runtime.
`ContextAssembler` zachowuje ranking, deduplikuje i przydziela osobne budżety
źródłom. Typowany `ContextPackV1` ma kompatybilny, jawnie niezaufany widok
`TurnContext.memories: list[str]`. Najlepszy hit nie wypada już przez podwójne
cięcie końca listy.

**B — failover `vector_memories`.** Gdy Qdrant wyłączony, padł albo zwrócił pusto, `MemoryStore.search_vector_memories()` robi cosine po JSON osi `semantic` w SQLite. Dual-write `semantic` przy `QDRANT_DUAL_WRITE=true`. Ingest Screenpipe przy padzie Qdranta jest fail-closed (`QdrantUnavailableError`) — nie wolno udawać „brak duplikatu”. Retrieval użytkownika może spaść na SQL; ingest nie.

**C — `remember` / `recall`.** `remember` i `remember_last_source` piszą tabelę
`memories` (plus indeks Qdrant przez `ManualMemoryService`, gdy działa).
Domyślny recall nadal używa Qdrant → B → podłańcuch tekstowy. Przy jawnym
`CONTEXT_TIMELINE_RECALL_ENABLED=true` i pytaniu z zakresem czasu najpierw
uruchamia się TimeFirstRetriever: FTS timeline'u → Screenpipe → semantic scout.
Pusty wynik tej ścieżki zachowuje filtr czasu i nie przechodzi do starszego
recall bez ograniczenia daty.

**D — Context Timeline V1 (opt-in).** `context_events` i `context_episodes` są
kanonicznym magazynem zdarzeń i epizodów. FTS5 używa BM25 oraz ograniczeń
czasu. Gdy dowodów jest mało, scout próbuje `semantic`, a następnie wybrane
pytaniem osie rezerwowe. Kandydat Qdrant musi mieć aktualny epizod SQL,
zgodną tożsamość i hash, żywe źródła ze zgodnymi wersjami oraz pasujący zakres
czasu. Treść i czas pochodzą z SQL; ranking zostaje w `retrieval_score`,
nie w `confidence`.

Kolekcja pamięci nadal może zawierać starsze typy wpisów. Scout odpytuje tę
samą kolekcję, ale pomija punkty bez kanonicznego epizodu i hashy źródeł.
`ContextEpisodeVectorizer` jest jawnym API; nie ma automatycznego wywołania
przy starcie. Dotychczasowy planer nadal odpytuje pięć osi. Liczba trafień
nadal steruje zatrzymaniem scouta; nie wdrożono jeszcze polityki rangi pytań.
Żadna tabela kontekstu nie steruje executorem.

Capabilities **nie** wchodzą do A/B/C. Osobna kolekcja, osobny RRF (normalizacja do stałego zestawu osi capability — inny wariant niż pamięć; nie scalać helperów).

### PLANNING — LLM proponuje, lokalny kod wiąże

**Pliki:** `model_router.py` (`ProposedPlan`, `ProposedStep`, `OpenAICompatiblePlanner`, `ModelRouter`), `assistant.py`.

`ProposedPlan`: `intent`, `response_text`, `confidence`, `requires_clarification`, `clarification_question`, `steps` (max 12). `ProposedStep.depends_on` to indeksy `int` wstecz. `extra="forbid"`.

Lokalne wiązanie (`OpenAICompatiblePlanner.plan`):

- JSON Schema `ProposedPlan`; jeden zły `action_id` albo zła krawędź `depends_on` **odrzuca cały plan** (`steps=[]`, clarification). Nie wycina się „zdrowych” kroków.
- Indeksy zależności mapowane na UUID `PlanStep.id`.
- Kontekst zewnętrzny (pamięć, ekran, web) jest **niezaufany**. Nie może stworzyć zadania ani obniżyć zgody. `tool_observations` nie idą do task-planera (pusta lista); idą tylko do ścieżki conversation.
- `response_text` nie wolno twierdzić, że akcja już się udała.
- `ModelRouter`: primary najpierw; fallback tylko przy niskiej pewności / niedostępności i gdy wolno chmurę. Conversation zostaje na primary.

Domyślne `Settings.llm_primary` w kodzie: `"local"`. Live stack często stawia Gemini — to konfiguracja, nie zmiana kontraktu. Qwen: fallback i digest Screenpipe. Nomic: tylko wektory.

### POLICY — allowlista i ryzyko wygrywają z modelem

**Pliki:** `actions.py` (`ActionSpec`, `ActionRegistry.enforce_policy`, `bind_execution_targets`), `routing/assembler.py` (`validate_plan`), `routing/validation.py`.

`ActionSpec`: `id`, `args_schema` (obiekt, bez `additionalProperties`), `risk`, `confirmation_required`, `handler`, `execution_layer` (1 system, 2 UIA, 3 UI.Vision), przykłady routingu.

`enforce_policy(step)`: nieznane `action_id` → wyjątek; argumenty vs schema; model **nie obniży** ryzyka ze speca; lokalny wymóg zgody albo `high` wymusza `confirmation_required`.

`bind_execution_targets`: tuż przed kolejką wiąże cel UIA / HWND dla
`hover_shell_item`, `open_shell_item`, `close_window_under_cursor` oraz akcji
aktywnego Notatnika. To nie jest nowa akcja z LLM.

44 `action_id` (kod `actions.py`):

`open_calendar`, `open_browser`, `open_url`, `open_folder`, `open_app`,
`hover_shell_item`, `open_shell_item`, `select_shell_folder`,
`select_shell_file`, `select_shell_items_by_extension`,
`select_shell_items_by_letter`, `select_listed_candidate`, `cursor_center`,
`cursor_return`, `snap_window_layout`, `open_chat`, `search_web`,
`open_gpt_chat`, `open_gemini_chat`, `describe_active_window`,
`minimize_active_window`, `minimize_all_windows`, `minimize_window_under_cursor`,
`close_window_under_cursor`, `copy_selected_text`, `copy_text_under_cursor`,
`copy_email_under_cursor`, `copy_number_under_cursor`,
`copy_sentence_under_cursor`, `select_sentence_under_cursor`,
`select_paragraph_under_cursor`, `rename_under_cursor`, `describe_text_target`,
`read_active_notepad`, `write_active_notepad`, `paste_text_safe`,
`describe_recent_activity`, `create_note`, `run_uivision_macro`, `remember`,
`remember_last_source`, `recall`, `list_capabilities`, `speak_text`.

Potwierdzenie (medium): `open_shell_item`, `close_window_under_cursor`, `rename_under_cursor`, `paste_text_safe`, `run_uivision_macro`, `remember`, `remember_last_source`. `create_note` jest medium bez zgody. High = 0.

`open_url` tylko http(s). `open_folder` / `open_app` = enum, nie dowolna ścieżka. UI.Vision = bazowa nazwa `.json` w runtime `macros`.

### EXECUTION — jedna kolejka, TTL, STOP

**Plik:** `listener/voiceloop/executor.py` (`CommandExecutor`).

- `asyncio.Queue` o `COMMAND_QUEUE_LIMIT=10`; overflow → `rejected`.
- Jeden plan naraz (`_current_execution`).
- `submit`: `bind_execution_targets` → `enforce_policy` na każdym kroku → clarification / puste kroki / pending confirmation / enqueue.
- `CONFIRMATION_TTL_SECONDS = 300`. Po restarcie plan można odtworzyć z SQLite; zgoda wygasa.
- Zależności: krok pomijany, gdy `depends_on` nie jest w `successful_steps`. Pierwszy `ActionResult.success=false` kończy plan jako `failed`.
- `stop_all()`: anuluje pending confirmation, opróżnia kolejkę, cancel bieżącego taska, `actions.stop()`, event `stop`.
- Statusy: `received` → `planning` → `awaiting_confirmation` | `queued` → `executing` → `succeeded`. Końce: `failed`, `cancelled`, `rejected`.

`SituationStateV1` jest ledgerem read-only. Executor wykonuje już zwalidowany `CommandPlan` i nie czyta tego stanu.

### OBSERVATION — skutek, nie nowa intencja

**Typy:** `ActionResult`, `ToolObservation`, `ScreenSnapshot`, `TurnContext`, `CommandView`, eventy SSE.

Właściciele: handler akcji zwraca `(message, data)` → `ActionResult`. Web/knowledge → `ToolObservation` (conversation only). Ekran → `ScreenSnapshot` przy `include_screen`. `TurnContext` to paczka dla planera, nie magazyn prawdy.

Nie wolno karmić outputu narzędzia z powrotem jako executable intent. Threshold Guard i Vectorscope są read-only wobec Qdrant.

`CommitmentItem` powstaje w eksperymentalnej analizie shadow. Może dać wiersz
do review przy jawnej fladze; nie jest potwierdzonym zobowiązaniem użytkownika.

---

## Kontrakty — kto czym jest

| Typ | Plik | Rola |
|---|---|---|
| `CommandRequest` | `models.py` | znormalizowane wejście |
| `TranscriptEnvelopeV1` | `models.py` | STT z pewnością i mówcami |
| `ProposedPlan` / `ProposedStep` | `model_router.py` | propozycja LLM; `depends_on: list[int]` |
| `CommandPlan` / `PlanStep` | `models.py` | plan do polityki i executora; `depends_on: list[str]` (id kroków) |
| `ActionSpec` | `actions.py` | jedyna wykonywalna zdolność |
| `ActionResult` | `models.py` | skutek kroku |
| `CommitmentItem` | `commitments/schema.py` | analiza zobowiązania; poza routingiem |
| `EvidenceItem` | `commitments/schema.py` | dowód commitmentu (`rule`/`vector`/`temporal`/`resolver`). **Nie** przyszły Evidence sytuacji |
| `StateProposal` | `situation/proposal.py` | propozycja stanu; nie jest `CommandPlan` i nie steruje planerem |
| `ResolutionDecisionV1` | `models.py` | werdykt V2 na subtask |
| `TurnContext` | `models.py` | kontekst jednej tury planera |
| `ContextEventV1` | `context/schema.py` | kanoniczna obserwacja na osi czasu |
| `ContextEpisodeV1` | `context/schema.py` | digest zakresu zdarzeń; jednostka wektoryzacji |
| `ContextPackV1` | `context/schema.py` | uporządkowany, niewykonywalny kontekst jednej tury |
| `ContextEntityV1` | `context/entities.py` | zatwierdzona osoba/projekt/tool/org ze stabilnym ID |

---

## `windows_shell.py`

Moduł lokalizuje i steruje **widocznymi** elementami pulpitu (`Progman` / `WorkerW`) oraz okien Eksploratora (`CabinetWClass` / `ExploreWClass`) przez UI Automation. „Aktywny folder” = okno Eksploratora **pod kursorem**, bez fallbacku na pulpit ani foreground.

Brak `subprocess`, `cmd.exe`, PowerShell. LLM nie dostaje narzędzia „uruchom polecenie”. Akcje `hover_shell_item` / `open_shell_item` / `select_shell_*` wołają ten locator. Nieporozumienie „shell = powłoka systemowa” jest błędem dokumentacyjnym.

---

## Documentation authority

Ten dokument i aktualny kod zastępują wcześniejsze handoffy, dzienne plany oraz
snapshoty operacyjne. Materiały pod `docs/archive/` służą wyłącznie historii
projektu i nie są kontraktem runtime.

---

## Następny spokojny krok

Context Timeline ma kontrakty, lokalne magazyny, adaptery jawne, selektywne
wektory, migrację pamięci i porównanie shadow. Następny krok to **prywatny
gold set puszczony przez `report-context-retrieval`**, nie automatyczne
włączenie workerów. Dopiero raport jakości może uzasadnić
`CONTEXT_TIMELINE_RECALL_ENABLED=true`.

Commitment shadow i opcjonalny wiersz review pozostają obserwacyjne. Nie
włączają SituationState ani executora. Rozszerzenie osi wymaga zatwierdzonego
słownika, przykładów i reguł nawigacji opisanych w dokumencie projektowym.
