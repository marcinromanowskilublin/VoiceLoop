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
| `context/evaluation.py` | harness retrievalu i porównanie shadow |
| `context/documents.py` | jawny odczyt dozwolonych plików tekstowych |
| `context/projection.py` | projekcja projektów i migracja pamięci SQL |
| `context/review.py` | żywy ekran deiktyczny i review zobowiązań |

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

### Potwierdzenie kandydatów semantycznych w SQLite

W ścieżce time-first Qdrant dostarcza kandydata, a SQLite dostarcza treść,
czas i provenance epizodu. Kandydat musi wskazywać istniejący `episode_id`,
zgodną parę `source`/`source_id` i bieżący `content_hash`. Wszystkie zdarzenia
źródłowe muszą istnieć, nie być usunięte ani wygasłe i odpowiadać wersjom
`source_event_hashes` zapisanym podczas indeksowania. Epizod również nie
może być usunięty ani wygasły. Zakres czasu sprawdzamy na epizodzie SQL,
według tej samej reguły przecinania przedziałów co przy odczycie FTS.

Wynik rankingu trafia wyłącznie do `retrieval_score`. `confidence` pozostaje
wartością kanonicznego elementu kontekstu; nie staje się prawdopodobieństwem
trafności obliczonym z RRF. Historyczny punkt bez odwołania do epizodu albo
wersji jego źródeł nie jest dowodem w tej ścieżce. Nowa weryfikacja wymaga
ponownego indeksowania takich epizodów z dostępem do SQLite; nie migruje ani
nie usuwa starych punktów automatycznie. Dotychczasowy odczyt pięciu osi poza
time-first pozostaje bez zmian.

Jeżeli włączony recall time-first zwróci pusty wynik dla wskazanego czasu,
zachowuje pustą odpowiedź i zakres czasu. Nie przechodzi do starszego
wyszukiwania bez filtra. Błędny, pozbawiony strefy lub leżący poza zakresem
timestamp wyniku Screenpipe jest pomijany, zamiast zastępowania go „teraz”.

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
CONTEXT_TIMELINE_AUTO_TTL_DAYS=14
CONTEXT_TIMELINE_WINDOWS_PROJECTION_ENABLED=false
CONTEXT_TIMELINE_DEICTIC_SCREEN_ENABLED=false
CONTEXT_TIMELINE_COMMITMENT_REVIEW_ENABLED=false
```

Parametry nie włączają background ingestu ani samplera foreground. Trzy nowe
flagi też zostają wyłączone: projekcja projektów Windows, zrzut ekranu dla
pytań deiktycznych oraz zapis review zobowiązań.

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

## Operacje

Adaptery uruchamia się jawnie z CLI. Żadna z tych komend nie startuje workera
i nie zmienia `CONTEXT_TIMELINE_RECALL_ENABLED`:

```powershell
cd listener
.venv\Scripts\python -m voiceloop.corpus ingest-context-documents --root ..\docs
.venv\Scripts\python -m voiceloop.corpus migrate-memories-to-timeline
.venv\Scripts\python -m voiceloop.corpus prune-context-timeline        # dry-run
.venv\Scripts\python -m voiceloop.corpus prune-context-timeline --apply
.venv\Scripts\python -m voiceloop.corpus report-context-retrieval --gold GOLD.jsonl
```

Komendy działają na bazie runtime, nie na korpusie, więc nie mają
`--data-root`; ścieżkę bazy wskazuje `--database`. Ingest dokumentów raportuje
przyczyny pominięcia: plik z sekretem jest odrzucany w całości, bo jedno
dopasowanie wzorca sugeruje kolejne w formach, których wzorce nie obejmują.
`report-context-retrieval` porównuje sam FTS z pełną kaskadą na prywatnym
zestawie gold i zawsze zwraca `recall_switch_allowed: false` — przełączenie
pozostaje decyzją operatora.

## Testy

```powershell
cd listener
.venv\Scripts\python -m pytest -c pyproject.toml -q `
  ..\tests\test_context_foundation.py `
  ..\tests\test_context_phase2.py `
  ..\tests\test_context_phase3.py
.venv\Scripts\python -m ruff check voiceloop ..\tests
```

Testy obejmują idempotencję, FTS, timeline, epizody, paginację Screenpipe,
scoping sesji, dry-run retencji, fail-closed Qdrant, encje z review gate, oba
magazyny spotkań, foreground, selektywne wektory, lazy reserve axis oraz metryki.

## Jawne adaptery

Te ścieżki istnieją w kodzie i nie startują same:

- `DocumentTimelineIngestor` czyta `.md`, `.txt`, `.rst` i `.markdown` tylko
  z podanych korzeni. Przycina katalogi generowane w trakcie przechodzenia
  drzewa, więc nie czyta `.git`. Pomija sekrety i pliki binarne. Nie
  wektoryzuje treści. Zdarzenie trzyma **digest i odnośnik**, nie drugą trwałą
  kopię pliku; `load_full_text()` czyta pełną treść na żądanie i ponownie
  sprawdza korzeń oraz rozszerzenie, więc podmieniony odnośnik nie wyprowadzi
  odczytu poza konfigurację.
- `project_windows_projects()` zapisuje nazwę projektu. Ścieżka zostaje poza
  tekstem zdarzenia. Domyślny watcher Windows jej nie woła.
- `MemoryTimelineMigrator` kopiuje istniejące `memories` idempotentnie.
  Wspomnienia ręczne nie dostają TTL. To nie jest migracja przy starcie.
- `question_is_deictic()` rozpoznaje „tutaj”, „to okno” i „na ekranie”. Samo
  słowo „to” nie uruchamia zrzutu. Pozycja ekranu jest żywa, ma TTL 300 s i
  nie trafia do Context Packu, dopóki flaga jest wyłączona — także wtedy, gdy
  żądanie ma `include_screen=true`.
- `commitment_review_event()` tworzy wiersz do review. Pole `executable` jest
  fałszywe. Nie awansuje zobowiązania i nie tworzy `action_id`.
- `compare_context_retrieval_shadow()` liczy różnicę metryk. Dodatnia delta
  nie włącza `CONTEXT_TIMELINE_RECALL_ENABLED`.

## Czas obserwacji i TTL adapterów

Oś czasu opisuje, kiedy coś zaszło, a nie kiedy skaner to zobaczył. Ma to
znaczenie, bo `content_hash` zdarzenia zawiera `started_at`, a upsert nadpisuje
ten czas:

| Adapter | `started_at` | TTL |
|---|---|---|
| Dokument (digest) | `st_mtime` pliku | 14 dni od skanu |
| Projekt Windows | `first_seen` rekordu | 14 dni od obserwacji |
| Review zobowiązania | czas wypowiedzi | 14 dni |
| Migrowana pamięć | `created_at` wpisu | brak |

Czas skanu żyje w metadanych (`observed_at`, `last_seen`). Gdyby `started_at`
brał `now()`, każdy przebieg przesuwałby całą znaną historię na teraz.
`CONTEXT_TIMELINE_AUTO_TTL_DAYS` domyślnie równa się horyzontowi wektorów.
Wiersze automatyczne wygasają i podlegają `prune_expired()`; jawne pamięci
użytkownika nie.

## Ograniczenia i dalszy rollout

- Brak automatycznego ingestu i pętli foreground.
- Brak automatycznej akceptacji encji osób.
- Prywatny gold set zostaje poza repo. Publiczny harness przyjmuje rekordy
  lokalnie i nie przełącza ruchu.
- Recall timeline pozostaje wyłączony do czasu raportu shadow i quality gate.
- Nowe tabele współistnieją z pamięcią A/B/C. Migracja jest wywołaniem jawnym.
