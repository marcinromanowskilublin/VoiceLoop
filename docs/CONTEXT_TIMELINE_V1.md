# Context Timeline V1

**VoiceLoop:** 0.3.0

Context Timeline jest lokalną, niewykonywalną warstwą dowodową VoiceLoop.
Czas jest osią główną; aplikacje, okna, osoby i projekty są metadanymi, a
embeddingi opisują znaczenie epizodów. Żaden element timeline'u nie może
utworzyć `action_id`, zastąpić zgody użytkownika ani potwierdzić wykonania akcji.

## Status i zakres

| Element | Stan | Domyślnie |
|---|---|---|
| Schematy i Context Pack | działa | aktywne |
| SQLite `context_events` / `context_episodes` | działa | aktywne po inicjalizacji DB |
| FTS5 zdarzeń i epizodów | działa z guardem | aktywne, gdy SQLite ma FTS5 |
| Historyczne wyszukiwanie Screenpipe | działa | wywołanie jawne |
| Ingest Screenpipe i spotkań | działa | brak pętli background |
| Foreground Win32 | działa jako pojedyncza próbka | brak pętli background |
| Rejestr encji i review kandydatów | działa | awans wyłącznie jawny |
| Wektory epizodów | działa | wywołanie jawne |
| Semantic scout i osie rezerwowe | działa | tylko lokalne backendy |
| Recall korzystający z timeline'u | eksperyment | `false` |
| Automatyczny ingest/prune/foreground | planowane | wyłączone |
| Produkcyjny quality gate | planowane | brak prywatnego gold setu w repo |

Flaga `CONTEXT_TIMELINE_RECALL_ENABLED=false` pozostaje domyślna. Po włączeniu
zmienia wyłącznie akcję `recall` zawierającą jawny zakres czasu. Nie uruchamia
workerów, nie zmienia routingu i nie omija polityki wykonania.

## Moduły

| Moduł | Odpowiedzialność |
|---|---|
| `context/schema.py` | kontrakty item/pack/event/episode |
| `context/assembler.py` | budżet, kolejność i deduplikacja Context Pack |
| `context/retrieval.py` | parser czasu, FTS, Screenpipe, semantic scout |
| `context/ingest.py` | jawny adapter Screenpipe → timeline |
| `context/meeting_ingest.py` | oba magazyny spotkań → timeline |
| `context/foreground.py` | pojedyncza obserwacja aktywnego okna Win32 |
| `context/entities.py` | identyfikatory, aliasy i resolver |
| `context/entity_registry.py` | trwałe encje i kolejka kandydatów |
| `context/vectorization.py` | selektywne wektory epizodów |
| `context/lifecycle.py` | fail-closed retencja SQL/FTS/Qdrant |
| `context/evaluation.py` | mierzalny harness retrievalu |

## Model danych

### `context_events`

Kanoniczna obserwacja źródłowa. Najważniejsze pola:

- `event_id`, `source`, `source_id`, `event_type`;
- `started_at`, `ended_at`;
- `app_name`, `process_name`, `window_title`;
- `is_foreground`, `foreground_confidence`, `focus_duration_ms`;
- `text`, `content_hash`, `metadata_json`;
- `sensitivity`, `expires_at`, `deleted_at`.

Idempotencję zapewnia `UNIQUE(source, source_id)`. `event_id` jest stabilnym
UUIDv5 i nie zawiera prywatnej treści. Indeks FTS obejmuje tekst, tytuł okna i
nazwę aplikacji.

### `context_episodes`

Epizod grupuje zdarzenia z ograniczonego zakresu czasu:

- zakres `started_at`–`ended_at`;
- `source_event_ids`;
- summary i tytuł;
- aplikacje oraz rozstrzygnięte `person_ids` / `project_ids`;
- lista faktycznie zapisanych `vector_spaces`;
- provenance, TTL i tombstone.

Deterministyczny fallback tworzy kubełki 10-minutowe. Spotkania używają granic
istniejących sesji. Model może później przygotować lepszy digest, ale nie
zmienia źródłowych zdarzeń.

### Context Pack

`ContextPackV1` jest typowany i zachowuje:

- źródło i stabilny `source_id`;
- zakres czasu;
- trust, confidence i sensitivity;
- przyczynę wyboru (`selection_reason`);
- score retrievalu i metadata.

`TurnContext.memories` pozostaje `list[str]` jako kompatybilny, jawnie
niezaufany widok. Jest to celowe zachowanie invariantu INV-04, nie dług
techniczny do „naprawienia” przez przekazanie modelowi obiektów wykonawczych.

## Przepływ zapisu

```text
Screenpipe / MeetingRecorder / Win32
  -> ContextEventV1
  -> SQLite context_events + FTS5
  -> grupowanie czasowe
  -> ContextEpisodeV1 + FTS5
  -> opcjonalny ContextEpisodeVectorizer
  -> Qdrant: semantic + tylko niepuste osie rezerwowe
```

Nie wektoryzujemy klatek OCR, nazw programów, HWND ani surowych ścieżek.
Domyślnie epizod dostaje `semantic`. `decision`, `intent`, `person_context` i
`topic` powstają wyłącznie wtedy, gdy istnieje odpowiadająca im jawna treść.

## Przepływ odczytu

```text
pytanie
  -> parser czasu (dziś / wczoraj / przedwczoraj / ostatnie N godzin lub dni)
  -> FTS5 epizodów i zdarzeń w zakresie
  -> historyczny Screenpipe, jeżeli lokalny timeline jest zbyt ubogi
  -> semantic scout, jeżeli nadal brakuje dowodu
  -> jedna właściwa oś rezerwowa, gdy pytanie jej wymaga
  -> ContextPackV1
  -> odpowiedź z provenance albo abstencja
```

Dla pytania z jawnym czasem semantic hit bez źródłowego timestampu jest
odrzucany. `created_at` punktu Qdrant nie udaje czasu obserwowanego zdarzenia.

## Encje i prywatność

`ContextEntityRegistry` rozdziela:

- zatwierdzone encje i ich aliasy;
- kandydatów oczekujących na review;
- status `approved` / `rejected`.

Kandydat wymaga co najmniej dwóch niezależnych `evidence_source_ids` oraz
jednego mocnego sygnału, zanim w ogóle może zostać zatwierdzony. Zatwierdzenie
jest wywołaniem jawnym. Ekstrakcja modelu i pojedynczy OCR nigdy nie awansują
osoby automatycznie.

`entity_id` jest UUIDv5 utworzonym z rodzaju i kanonicznej nazwy. Widoczna
etykieta oraz aliasy są oddzielnymi polami i mogą zostać skorygowane bez
przepisywania wszystkich odwołań.

## Foreground

Screenpipe nie dostarcza wiarygodnego `is_foreground` ani HWND. Jego zdarzenia
zapisują `foreground_confidence=unknown`. Jawny `ForegroundSampler` korzysta z
Win32 i tworzy osobne zdarzenie `win32_foreground` z confidence `observed`.
Kod odrzuca kombinację znanego `is_foreground` i confidence `unknown`.

Sampler nie ma automatycznej pętli. Dzięki temu samo zaimportowanie modułu nie
uruchamia monitorowania pulpitu.

## Retencja i usuwanie

`ContextLifecycleService.prune_expired()` jest dry-run domyślnie:

1. wylicza wygasłe rekordy;
2. przy jawnym `dry_run=false` usuwa powiązane punkty Qdrant;
3. dopiero po sukcesie Qdranta zakłada tombstone w SQLite;
4. usuwa projekcję FTS;
5. pozostawia rekord kanoniczny do audytu.

Awaria Qdranta zatrzymuje lokalną część operacji. Niedostępny magazyn nie jest
traktowany jak „punktu nie ma”.

## Konfiguracja

```env
CONTEXT_TIMELINE_RECALL_ENABLED=false
CONTEXT_TIMELINE_BUCKET_MINUTES=10
CONTEXT_TIMELINE_MAX_RESULTS=500
```

Parametry nie włączają background ingestu ani samplera foreground.

## Ewaluacja

`evaluate_context_retrieval()` mierzy:

- Recall@k;
- Mean Reciprocal Rank;
- poprawność abstencji;
- kompletność provenance;
- pokrycie oczekiwanych osi rezerwowych.

Publiczne repo zawiera harness i testy syntetyczne. Prywatne pytania, nazwy osób
i realne treści Screenpipe nie należą do repo.

## Failure semantics

- brak FTS5 → kontrolowany fallback SQL `LIKE`;
- błąd Screenpipe → recall może kontynuować na lokalnym timeline i Qdrancie;
- błąd embeddingu/Qdranta przy odczycie → brak semantic scout, bez wyjątku do
  wykonania akcji;
- błąd Qdranta przy usuwaniu → brak lokalnego tombstone;
- brak timestampu w wektorze przy pytaniu czasowym → odrzucenie hitu;
- nierozstrzygnięta osoba → literalne FTS, bez wymuszonego `person_id`.

## Testy

```powershell
cd listener
.venv\Scripts\python -m pytest -c pyproject.toml -q `
  ..\tests\test_context_foundation.py `
  ..\tests\test_context_phase2.py
.venv\Scripts\python -m ruff check voiceloop ..\tests
```

Testy obejmują idempotencję, FTS, timeline, epizody, paginację Screenpipe,
scoping sesji, dry-run retencji, fail-closed Qdrant, encje z review gate, oba
magazyny spotkań, foreground, selektywne wektory, lazy reserve axis oraz metryki.

## Ograniczenia i dalszy rollout

- Brak automatycznego ingestu i pętli foreground.
- Brak adapterów dokumentów oraz `WindowsContextService`.
- Brak automatycznej akceptacji encji osób.
- Brak opublikowanego prywatnego gold setu.
- Recall timeline pozostaje wyłączony do czasu raportu shadow i quality gate.
- Nie ma migracji big-bang starych `memories`; nowe tabele współistnieją z
  pamięcią A/B/C.
