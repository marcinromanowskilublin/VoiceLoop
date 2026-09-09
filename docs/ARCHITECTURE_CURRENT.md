# VoiceLoop — aktualny baseline architektury

**ETAP 0.** Stan kodu z 7 września 2026.  
**Źródło prawdy:** `listener/voiceloop/`. Starsze opisy ustępują kodowi.  
**Ten plik nie zmienia zachowania produkcyjnej pętli.** ETAP 1–3 dodały kontrakty i testy; planer nadal nie wykonuje i nie czyta stanu jako intencji.

Rdzeń zostaje:

```text
planer → walidacja → allowlista → polityka → executor
```

Brak jawnego modelu sytuacji to **PLANOWANE**, nie dziura w executorze.

---

## Trzy półki — nie mieszać

### DZIAŁA (produkcyjna pętla)

- Wejście → `CommandRequest` → zapis SQLite + dedupe → STOP/pauza → tanie reguły → (opcjonalnie) n8n → plan LLM albo plan V1 → `enforce_policy` → jedna kolejka executora → `ActionResult`.
- LLM zwraca `ProposedPlan`. Lokalny kod wiąże go do `CommandPlan`. Model nie wykonuje akcji i nie dostaje powłoki.
- Allowlista: 42 `ActionSpec` w `listener/voiceloop/actions.py`. 0 akcji `high`. 8 `medium`, z czego 7 wymaga potwierdzenia.
- `windows_shell.py` = UIA pulpitu i Eksploratora. To nie jest spawn `cmd` / PowerShell dla LLM.
- Pamięć jest hybrydą SQL + Qdrant w trzech rolach (A/B/C poniżej). To nie jest BM25 i nie jest „brakiem hybrydy”.
- Capabilities żyją w osobnej kolekcji Qdrant `voiceloop_capabilities_v1`. Nie merge z `voiceloop_memory`.
- Threshold Guard mierzy progi. Nie zapisuje punktów i nie zastępuje retrievalu.
- `TranscriptEnvelopeV1` jest kontraktem STT. Routing V2 jest w kodzie, ale **nie steruje** produkcją przy domyślnych flagach.

### EKSPERYMENT (kod jest, nie steruje produkcją albo nie jest wpięty)

- Routing V2: `routing_v2_enabled=true`, `shadow_mode=true`, `routing_v2_execute=false`. Liczy się obok V1. Live wymaga quality gate + zgodnego fingerprintu + (opcjonalnie) canary.
- Commitment Layer: detector + scoring + schema, testy w `tests/test_commitment_analysis.py`. Analiza tekstu only. **Nie** importuje jej `assistant.py` ani `app.py`.
- `EvidenceItem` w `commitments/schema.py` (`rule` / `vector` / `temporal` / `resolver`) — to **obecny** typ commitmentów. Detector emituje wyłącznie `kind="rule"`. To **nie** jest przyszły `EvidenceItemV1` sytuacji.
- n8n: `n8n_enabled=false`. Webhook dokładnych fraz, bez Execute Command. Pad n8n nie wali asystenta.
- Hume EVI: szkielet, domyślnie off.
- `hybrid` w `voice_conversation.py` = barge-in / reuse strumienia Deepgram, **nie** retrieval.
- Shadow Qdrant (`qdrant_memory_next_collection`) i kalibracja V2 (`routing_v2_calibration_mode=off`) — pomiar, nie sterowanie.
- `EvidenceItemV1` (`situation/evidence.py`) — kontrakt dowodu; nie jest commitment `EvidenceItem` i nie wiąże akcji.
- `SituationStateV1` — ledger in-memory + `GET /api/v1/situation`. Planer nie czyta go jako intencji. LLM nie pisze. Brak tabeli SQL `situation`.
- `StateProposal` (`situation/proposal.py`) — model może proponować (`propose_fact` + `evidence_refs`). Lokalne `StatePolicy` + `StateReducer` decydują i wołają `append_event(actor=local_code)`. **Nie** wpięte w planer / `assistant.py`. Tylko testy / przyszły shadow.

### PLANOWANE (nie wpinąć w planer w tym etapie)

- Wpięcie commitment analyzer w produkcyjny stan (shadow = ETAP 5).
- Wpięcie `SituationStateV1` w routing / executor jako źródło akcji — **zakazane**.
- Etapy wektorowe i temporalne commitmentów (schema ma sloty, kod ich nie wypełnia).
- Live Routing V2 po quality gate z liczbami.
- Automatyczny prune wektorów (`VECTOR_MEMORY_PRUNE_ENABLED=false`).

Kontrakty ETAP 1–3 są w repo i **nie sterują** planerem:

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

  Commitment Layer ──► tylko testy tekstu, poza tą strzałką
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

### MEMORY — hybryda A/B/C, nie BM25

**Pliki:** `memory.py`, `qdrant_memory.py`, `memory_vectorization.py`, `manual_memory.py`, `assistant.py` (`_vector_memories_for_request`, `_create_plan`), `actions.py` (`remember` / `recall`), `capability_index.py`.

Dwie kolekcje Qdrant, **osobno**:

| Kolekcja | Osie | Po co |
|---|---|---|
| `voiceloop_memory` | `semantic` 0.40, `topic` 0.20, `intent` 0.15, `decision` 0.15, `person_context` 0.10 | kontekst planera i recall wektorowy |
| `voiceloop_capabilities_v1` | `semantic`, `intent`, `target_context` | katalog „co potrafisz” / V2 |

RRF pamięci: `k=60`, `VECTOR_MEMORY_MIN_SCORE=0.0` (celowo — fusion jest na rangach). Schema dokumentów: `memory-documents-v2`. Embeddingi: lokalny Nomic 768d przez LM Studio. Handoff §10 i Notion obowiązują dla wektorów; nie zastępować ich promptem z §2.

SQL (`data/voiceloop.db`, WAL) jest **partnerem** Qdranta, nie zamiennikiem BM25:

**A — concat do planera.** `AssistantService._create_plan` zbiera równolegle: hitów Qdrant (albo fallback B) **oraz** `memory.list_memories()`. Jeśli wśród wektorów nie ma `source=manual_memory`, dopina 3 albo 30 jawnych wspomnień. Do tego ostatnie streszczenia akcji. Całość idzie do `TurnContext.memories` → LLM. To jest hybryda kontekstowa, nie jeden ranking BM25+wektor.

**B — failover `vector_memories`.** Gdy Qdrant wyłączony, padł albo zwrócił pusto, `MemoryStore.search_vector_memories()` robi cosine po JSON osi `semantic` w SQLite. Dual-write `semantic` przy `QDRANT_DUAL_WRITE=true`. Ingest Screenpipe przy padzie Qdranta jest fail-closed (`QdrantUnavailableError`) — nie wolno udawać „brak duplikatu”. Retrieval użytkownika może spaść na SQL; ingest nie.

**C — `remember` / `recall`.** `remember` i `remember_last_source` piszą tabelę `memories` (plus indeks Qdrant przez `ManualMemoryService`, gdy działa). `recall` najpierw Qdrant 5 osi, potem B, na końcu **podłańcuch tekstowy** po `memories` (`retrieval: lexical_fallback`). To nie jest BM25.

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

`bind_execution_targets`: tuż przed kolejką wiąże cel UIA / HWND dla `hover_shell_item`, `open_shell_item`, `close_window_under_cursor`. To nie jest nowa akcja z LLM.

42 `action_id` (kod `actions.py`, nie tabela z handoffu §11):

`open_calendar`, `open_browser`, `open_url`, `open_folder`, `open_app`, `hover_shell_item`, `open_shell_item`, `select_shell_folder`, `select_shell_file`, `select_shell_items_by_extension`, `select_shell_items_by_letter`, `select_listed_candidate`, `cursor_center`, `cursor_return`, `snap_window_layout`, `open_chat`, `search_web`, `open_gpt_chat`, `open_gemini_chat`, `describe_active_window`, `minimize_active_window`, `minimize_all_windows`, `minimize_window_under_cursor`, `close_window_under_cursor`, `copy_selected_text`, `copy_text_under_cursor`, `copy_email_under_cursor`, `copy_number_under_cursor`, `copy_sentence_under_cursor`, `select_sentence_under_cursor`, `select_paragraph_under_cursor`, `rename_under_cursor`, `describe_text_target`, `paste_text_safe`, `describe_recent_activity`, `create_note`, `run_uivision_macro`, `remember`, `remember_last_source`, `recall`, `list_capabilities`, `speak_text`.

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

`CommitmentItem` **nie** jest obserwacją runtime. Żyje tylko w module eksperymentalnym.

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

---

## `windows_shell.py`

Moduł lokalizuje i steruje **widocznymi** elementami pulpitu (`Progman` / `WorkerW`) oraz okien Eksploratora (`CabinetWClass` / `ExploreWClass`) przez UI Automation. „Aktywny folder” = okno Eksploratora **pod kursorem**, bez fallbacku na pulpit ani foreground.

Brak `subprocess`, `cmd.exe`, PowerShell. LLM nie dostaje narzędzia „uruchom polecenie”. Akcje `hover_shell_item` / `open_shell_item` / `select_shell_*` wołają ten locator. Nieporozumienie „shell = powłoka systemowa” jest błędem dokumentacyjnym.

---

## Handoff 11.08 vs kod

**§2 (prompt startowy) — częściowo nieaktualny.** Traktuje n8n jako stały pierwszy router, pomija V2 shadow, capabilities, `TranscriptEnvelopeV1`, hybryde A/B/C i to, że `POST /stop` jest soft barge-in. Tabelę akcji z §11 zastępuje `actions.py` (42, nie ~13).

**§10 i Notion (30.08) — obowiązują dla wektorów:** pięć osi, wagi, RRF, dual-write `semantic`, SQLite jako failover, osobna kolekcja capabilities, `VECTOR_MEMORY_MIN_SCORE=0.0`, fail-closed ingest. Notion ma już przestarzale liczby allowlisty (26+7); wygrywa kod.

---

## Następny spokojny krok

ETAP 1–4 (invariants, `EvidenceItemV1`, `SituationStateV1` read-only, `StateProposal` poza planerem) są zrobione. Następny spokojny krok to **ETAP 5: commitment shadow** — nie wpinąć analizatora w produkcyjny stan w tym samym przebiegu.
