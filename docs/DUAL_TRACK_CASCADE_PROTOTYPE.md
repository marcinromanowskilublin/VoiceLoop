# Prototyp dual-track: RRF5 kontra kaskada 2+2+1

**Status:** eksperyment shadow, domyślnie wyłączony.
**Granica:** wynik nigdy nie zmienia odpowiedzi, planu, akcji ani potwierdzenia.

Prototyp odpowiada na jedno pytanie techniczne: czy ograniczona kaskada odzyskuje
te same kanoniczne epizody co obecny pięcioosiowy Weighted RRF, przy mniejszej
liczbie prób osi. Nie jest mechanizmem awansu kaskady do produkcji.

## Architektura

```text
czas z zapytania ──► wspólny bounded TimeFirstQueryPlan
                           │
                 ┌─────────┴─────────┐
                 │                   │
        baseline RRF5         kaskada 2+2+1
        pięć osi naraz        maks. pięć osi falami
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 SQL/hash/TTL/range gate
                           ▼
              arbiter agreement/conflict/uncertain
                           ▼
             content-free trace z TTL; brak serving
```

Oba tory czytają tę samą izolowaną kolekcję
`voiceloop_context_episodes_v1`. Nie czytają aktywnej kolekcji
`voiceloop_memory`. Kod odrzuca konfigurację, w której obie nazwy są takie
same.

## Jeden parser czasu i tekstu wyszukiwania

`normalize_context_text`, `build_context_search_text` i
`build_time_first_query` są jednym źródłem prawdy dla normalizacji, usuwania
rusztowania czasowego z FTS oraz budowania zakresu. Analiza zdania shadow nie
składa osobnego `search_text`: ponownie używa kanonicznego planu, a nadpisuje
wyłącznie zweryfikowane role czasu i preferowane osie.

Obsługiwane polskie formy obejmują:

- `wczoraj`, `przedwczoraj`, `dziś` i `dzisiaj`;
- ostatnie okno z liczbą lub liczebnikiem, np. `ostatnie 2 godziny`,
  `ostatnie dwie godziny` i `ostatnie trzy miesiące`;
- `kilka` / `kilku` / `parę` / `paru` dni, tygodni albo miesięcy temu;
- liczbę cyfrą lub słowem, np. `2 miesiące temu`, `dwa tygodnie temu` i
  `dwa lata temu`;
- warianty `w zeszłym`, `ostatnim`, `poprzednim` lub `tamtym`
  tygodniu, miesiącu i roku oraz formy dopełniacza tygodnia i miesiąca;
- `około miesiąca temu` i `pół roku temu`.

Formy przybliżone tworzą skończony przedział `start`–`end`, precyzję
`approximate`, relację i jawnie niższą pewność. Nieznany czas pozostaje
`unbounded`; oba tory odrzucają taki pomiar jako `unbounded_rejected`, zanim
wykonają wyszukiwanie wektorowe.

## Jawny, idempotentny indexer

Indexer nie działa przy starcie aplikacji i nie jest workerem. Domyślnie
wykonuje wyłącznie podgląd SQL:

```powershell
cd listener
.\.venv\Scripts\python.exe -m voiceloop.corpus index-context-episodes
```

Podgląd nie łączy się z Qdrant, więc `would_index` oznacza liczbę rozważanych
żywych epizodów, a nie różnicę względem zdalnej kolekcji. Jawny zapis:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus index-context-episodes --apply
```

Przy `--apply` indexer:

1. wymaga lokalnego endpointu embeddingów i lokalnego Qdrant;
2. sprawdza punkt po stabilnym `source/source_id`, `content_hash` oraz
   fingerprintcie obejmującym TTL, dokumenty osi i wersje źródeł;
3. pomija niezmieniony punkt przed wywołaniem embeddingu;
4. zapisuje `episode_id`, hash epizodu i hashe wszystkich źródeł;
5. aktualizuje w SQLite wyłącznie listę rzeczywiście zapisanych przestrzeni.

Awaria Qdrant nie jest traktowana jako brak punktu. Przebieg można bezpiecznie
powtórzyć po awarii.

## Tory i arbiter

Baseline osadza pięć dokumentów zapytania w jednym batchu i używa istniejącego
Weighted RRF. Kaskada odpytuje maksymalnie dwie osie w jednej fali. Pierwszą
osią jest `semantic`, a drugą jawnie wskazana pytaniem oś rezerwowa albo
deterministyczny następny wymiar.

Każdy hit obu torów przechodzi te same bramki:

- istniejący i żywy epizod SQLite;
- zgodne `source`, `source_id` i `content_hash`;
- żywe zdarzenia źródłowe z tymi samymi hashami;
- przecięcie kanonicznego czasu epizodu z zakresem pytania.

Arbiter nie wybiera wyniku do odpowiedzi:

| Warunek | Werdykt |
|---|---|
| Oba tory mają ten sam pierwszy kanoniczny epizod | `agreement` |
| Oba mają wyniki, ale zbiory są rozłączne | `conflict` |
| Jeden tor jest pusty, oba są puste albo zbiory częściowo się pokrywają z innym top-1 | `uncertain` |

Brak rozpoznanego zakresu czasu daje `unbounded_rejected`. Limit współbieżności
odrzuca nadmiar jako `concurrency_limit`; nie tworzy nieograniczonej kolejki.

## Trwały trace bez treści

Tabela `context_dual_track_traces` nie przechowuje pytania, odpowiedzi, tytułu,
treści epizodu, jawnego `request_id`, `session_id` ani `source_id`. Zapisuje:

- skrót requestu i skróty wyników solone identyfikatorem trace’u;
- werdykt, powód, liczności i overlap;
- precyzję i relację czasu oraz content-free rolę epistemiczną i bramkę
  relewancji;
- osie próbowane i zakończone, fale oraz powody stopu;
- liczbę odrzuceń przez SQL gate;
- czasy obu torów, timeout i klasę błędu bez komunikatu.

Trace ma TTL konfigurowany przez
`CONTEXT_DUAL_TRACK_TRACE_RETENTION_DAYS` i jest objęty workerem retencji.

Role czasowe zdania mają jednego właściciela zapisu. Gdy cascade shadow jest
włączony i ma dostępne providery, zapisuje role przed rozpoczęciem retrieval,
więc timeout albo błąd Qdrant ich nie gubi; uruchamiany równolegle dual-track
dostaje wtedy `persist_statement=False` i nie wykonuje duplikatu. Gdy cascade
nie jest dostępny, `DualTrackRunner` zachowuje domyślne
`persist_statement=True`, co utrzymuje samodzielne użycie runnera.

## Konfiguracja runtime

```env
QDRANT_CONTEXT_EPISODE_COLLECTION=voiceloop_context_episodes_v1
CONTEXT_DUAL_TRACK_ENABLED=false
CONTEXT_DUAL_TRACK_MAX_INFLIGHT=1
CONTEXT_DUAL_TRACK_TIMEOUT_SECONDS=5
CONTEXT_DUAL_TRACK_TRACE_RETENTION_DAYS=30
```

Włączenie flagi dodaje pomiar wyłącznie dla pytań z rozpoznanym czasem i bez
kroków wykonawczych. Task startuje dopiero po utworzeniu oraz przyjęciu planu,
więc nie konkuruje z modelem podczas budowania bieżącej odpowiedzi. Ścieżka
odpowiedzi nadal korzysta z dotychczasowego kontekstu i routingu.

## Raport CLI i gold set

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus report-context-retrieval `
  --gold ..\tests\fixtures\context-dual-track-gold-v1.jsonl `
  --eval-now 2026-10-19T12:00:00+00:00
```

Raport uruchamia faktyczny RRF5 i 2+2+1 na izolowanej kolekcji. Zawiera metryki
obu torów, delty, rozkład werdyktów, zgodność z oczekiwanym werdyktem i zdrowie
providerów. Pole `promotion_allowed` ma zawsze wartość `false`.

Repo zawiera 40 przykładów syntetycznych: 24 oczekiwane `agreement`, osiem
`conflict` i osiem `uncertain`. Sprawdzają format, klasy pytań, deterministyczny
czas i raportowanie; nie dowodzą jakości prywatnej pamięci. Prywatny gold set
pozostaje poza repo.

## Demonstrator read-only

Chronione tokenem endpointy:

- `GET /api/v1/context/dual-track/status` — tryb, rzeczywista liczba punktów
  kolekcji, pomocnicze znaczniki SQL, ich zgodność i content-free agregaty:
  liczności według werdyktu, precyzji czasu, bramki relewancji i roli
  epistemicznej, średnie czasy, timeouty oraz czas ostatniej próbki;
- `GET /api/v1/context/dual-track/traces` — ostatnie content-free trace’y.

Panel pokazuje te dane pod etykietą `PROTOTYPE / READ-ONLY`. Nie istnieje
endpoint `POST`, który włącza flagę, uruchamia indexer, generuje zapytanie lub
zmienia odpowiedź.

## Weryfikacja i ograniczenia

Testy obejmują izolację kolekcji, idempotencję, trzy werdykty arbitra, timeout,
limit współbieżności, trwałość i brak treści trace’u, integralność SQL,
40-elementowy gold, raport CLI, ochronę tokenem i brak metod zapisu w API.

Testy nie dowodzą jakości prywatnych danych ani wydajności docelowego Windows.
Kroki wymagające człowieka są prowadzone osobno w
[`DO_WYKONANIA_PRZEZ_CZLOWIEKA.md`](DO_WYKONANIA_PRZEZ_CZLOWIEKA.md).
