# Dynamic Memory Coupling V1 — plan inżynierski i stan wdrożenia

**Status:** fundament techniczny wdrożony w trybie shadow; brak przełączenia
odpowiedzi live.  
**Zakres:** SQLite/FTS, izolowany indeks epizodów Qdrant, wersjonowanie indeksu,
adaptacyjna polityka wyszukiwania, ewaluacja i bezpieczny rollout.  
**Niezmienna granica:** pamięć pozostaje kontekstem. Nie tworzy `action_id`,
nie obniża ryzyka i nie zastępuje potwierdzenia.

## Werdykt po rewizji

Kierunek jest technicznie sensowny, przyszłościowy i rozwijalny pod trzema
warunkami:

1. SQLite pozostaje jedynym kanonem treści, czasu, wersji i retencji.
2. Qdrant pozostaje odbudowywalnym indeksem kandydatów, a każdy wynik
   semantyczny jest hydratowany i weryfikowany w SQLite.
3. Adaptacja wybiera koszt i strategię retrievalu, ale nie zmienia prawdy ani
   uprawnień wykonawczych.

Obecne RRF5 i 2+2+1 nie są architekturą docelową. Są strategiami, które można
porównywać i wymieniać za wspólnymi kontraktami.

## Co działało przed tą rewizją

- kanoniczne `context_events` i `context_episodes` w SQLite;
- FTS5/BM25, przedziały czasu, TTL i tombstone;
- pięć named vectors: `semantic`, `topic`, `intent`, `decision`,
  `person_context`;
- SQL/hash/TTL/range gate w
  `TimeFirstRetriever._verified_episode_item()`;
- izolowany, jawny indexer epizodów;
- shadow RRF5 kontra 2+2+1, timeout, limit współbieżności i content-free trace;
- publiczny, syntetyczny pakiet dowodów.

To potwierdzało działanie kontraktu technicznego. Nie potwierdzało
adaptacyjności, jakości prywatnych danych ani gotowości live.

## Problemy znalezione w audycie

1. Time-first recall dostawał główną kolekcję `voiceloop_memory`, mimo że
   indexer epizodów zapisuje do `voiceloop_context_episodes_v1`.
2. `vector_spaces_json` był tylko znacznikiem; nie było desired/applied
   revision ani widocznego stanu błędu.
3. Fingerprint indeksu nie obejmował modelu embeddingowego i polityki prefiksu.
4. Adaptery mogły ustawić `metadata.vectorize=false`, ale indexer tego nie
   respektował.
5. Kaskada zatrzymywała się głównie po liczbie wyników. Pierwsza oś mogła
   zapełnić limit i zagłuszyć drugą oś tej samej fali.
6. `Recall@k` był faktycznie HitRate@k: jedno trafienie wystarczało nawet przy
   wielu oczekiwanych źródłach.
7. Równość liczników SQL i Qdrant była pokazywana jako „consistent”, mimo że
   nie sprawdzała rewizji konkretnych rekordów.

## Fundament wdrożony w tej rewizji

### 1. Transakcyjne sprzężenie SQL → indeks

Nowe tabele:

- `context_index_state` — desired operation/revision, applied revision,
  fingerprint, profil embeddingu, osie, status, próby i ostatnia klasa błędu;
- `context_index_outbox` — idempotentne zadania `upsert` / `delete`.

Upsert albo tombstone epizodu zapisuje stan indeksu i outbox w tej samej
transakcji SQLite. Indexer:

- pracuje at-least-once i idempotentnie;
- nie uznaje awarii Qdrant za pustą kolekcję;
- odrzuca spóźnione zakończenie starej rewizji;
- oznacza błąd i pozwala na ponowienie;
- wykrywa zmianę modelu embeddingowego przez fingerprint;
- respektuje `vectorize=false`;
- aktualizuje `vector_spaces_json` dopiero po zgodnym zapisie indeksu.

Nie uruchomiono automatycznego workera. Przetwarzanie pozostaje jawne przez
`index-context-episodes --apply`.

### 2. Właściwa kolekcja time-first

`ActionRegistry` ma osobną zależność `context_qdrant`. W runtime dostaje
`voiceloop_context_episodes_v1`; stary recall i pamięć planera zachowują główną
kolekcję. Dzięki temu semantic scout timeline'u widzi punkty tworzone przez
indexer epizodów.

### 3. Adaptacyjna polityka per zapytanie

`RetrievalPolicyV1` jest deterministycznym i wersjonowanym kontraktem. Dla
jednego zapytania określa:

- domenę: `general`, `windows` albo `rpg`;
- klasę: factual, operational, creative albo exploratory;
- ryzyko;
- przewagę FTS i wektorów;
- wagi osi, preferowane osie, `qmax`, wielkość i liczbę fal;
- minimalną liczbę dowodów i minimalne pokrycie osi;
- budżet kandydatów i warunek ekspansji.

Domena jest zaufaną wskazówką wejściową, nie wnioskiem z pobranej treści.
Każda polityka ma `require_canonical_sql=true` i
`serving_eligible=false`.

Przykładowe zachowanie:

- Windows + dokładna operacja/ścieżka: exact-first, bez kosztu wektorów;
- Windows + pytanie architektoniczne: większy udział semantyki, ale nadal
  wąski budżet;
- RPG + pytanie o kanon: FTS/SQL-first;
- RPG + generowanie motywu: szersza kaskada, większa różnorodność i co
  najmniej dwie osie, nadal bez automatycznego zapisu jako fakt.

### 4. Adaptacyjny retriever shadow

`AdaptiveHybridRetriever` wykonuje:

```text
bounded query
  -> SQLite/FTS i opcjonalny Screenpipe
  -> ocena wystarczalności według RetrievalPolicyV1
  -> wybrane fale named vectors
  -> weryfikacja kandydatów w SQLite
  -> ważona fuzja rang FTS + vectors
  -> ContextPackV1 + content-free trace
```

W adaptacyjnej kaskadzie trafienia dwóch osi jednej fali są najpierw scalane.
Pierwsza oś nie może już zużyć całego limitu przed oceną drugiej.

Brak czasu nadal blokuje ekspansję wektorową w tym etapie. To świadomy guard:
nie ma jeszcze zatwierdzonego, trwałego scope domeny dla unbounded RPG.

### 5. Poprawione metryki

Ewaluacja raportuje teraz osobno:

- prawdziwy source Recall@k;
- HitRate@k;
- Precision@k;
- nDCG@k;
- MRR tylko dla zapytań pozytywnych;
- accuracy abstencji na przykładach negatywnych;
- answerability accuracy i false-positive rate;
- provenance wyłącznie dla faktycznie zwróconych wyników.

Nowa komenda:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus report-adaptive-context-retrieval `
  --gold <prywatny-gold.jsonl> `
  --database ..\data\voiceloop.db `
  --domain windows `
  --eval-now 2026-11-16T20:00:00+01:00
```

Wynik zawsze zawiera `promotion_allowed=false`.

## Co pozostaje poza wdrożeniem

To są blokery przed live recall, nie kosmetyka:

1. Produkcyjny planer nadal korzysta z legacy
   `AssistantService._vector_memories_for_request()`, który może czytać treść
   payloadu głównej kolekcji. Nie został przełączony na nowy retriever.
2. Domena jest dziś przekazywana jawnie do biblioteki/CLI. Nie ma jeszcze
   zaufanego mapowania aktywny program/projekt/postać → domena ani polityki
   per podzapytanie.
3. Unbounded semantic search jest celowo zablokowany.
4. Outbox nie ma automatycznego workera. Jawny sweep indexera wykrywa także
   zmianę źródeł i modelu, ale nie ma jeszcze ciągłego przetwarzania.
5. Retencja timeline'u i ogólny worker private-data mają dwa różne przepływy.
   Przed live trzeba wskazać jednego właściciela semantyki delete/tombstone.
6. Brakuje prywatnego gold setu z prawdziwym lokalnym modelem embeddingowym.
7. Nie wykonano jeszcze pomiaru P95/P99, CPU, RAM i blokad SQLite na docelowym
   Windowsie.

## Plan dalszego rollout'u

### Etap A — obecny PR: fundament shadow

Warunek ukończenia:

- testy rewizji, błędów, retry i spóźnionego joba;
- testy kontrastowe Windows/RPG;
- kanoniczna hydracja każdego wyniku adaptacyjnego;
- brak wpływu na odpowiedź i akcje;
- pełny test suite oraz Ruff.

Rollback: wycofanie gałęzi. Ścieżka live nie została przełączona.

### Etap B — prywatna ewaluacja na tym komputerze

Wymagane:

- osobne lub jawnie oznaczone przykłady `general`, `windows`, `rpg-canon`,
  `rpg-creative`;
- wiele oczekiwanych źródeł, negatywy, konflikty i pytania mieszane;
- realny lokalny embedding zamiast lexical-hash;
- pomiar FTS-only, semantic-only, RRF5, stałego 2+2+1 i adaptive.

Proponowane twarde bramki:

- 100% serwowanych kandydatów ma zgodną rewizję SQL;
- zero wpływu shadow na plan i wykonanie;
- false-positive rate nie gorszy niż baseline;
- source Recall@k i nDCG@k nie gorsze osobno dla każdej domeny;
- dynamiczna polityka zużywa mniej osi/zapytań niż RRF5 przy wyniku
  non-inferior;
- P95 mieści się w budżecie zatwierdzonym przez operatora.

Dokładne wartości non-inferiority i P95 wymagają decyzji użytkownika po
obejrzeniu wyników.

### Etap C — zaufany scope i podzapytania

Do wdrożenia:

- `RetrievalSubqueryV1` z domeną, zakresem czasu, filtrami, ryzykiem i
  wspólnym budżetem;
- zaufane źródło domeny: konfiguracja projektu, aktywny program lub jawne
  wskazanie użytkownika;
- histereza na poziomie programu/postaci/projektu;
- możliwość różnych polityk dla kilku podzapytań jednej wypowiedzi;
- filtr domeny w kanonicznym SQL i payloadzie Qdrant;
- test cross-domain leakage.

Nie wolno pozwolić, aby tekst pobrany z pamięci przełączał własną domenę lub
rozszerzał budżet.

### Etap D — automatyczny index worker

Do wdrożenia:

- mały worker outbox z limitem współbieżności;
- lease/claim, retry z backoffem i dead-letter;
- desired/applied revision oraz ignorowanie starego joba;
- telemetry: outbox lag, failed, stale, orphan i missing;
- rebuild całej generacji indeksu bez blokowania SQLite;
- jedna semantyka retencji.

Rollback: zatrzymanie workera. SQLite/FTS działa dalej.

### Etap E — canary recall

- tylko recall/context, nigdy action selection;
- mała, jawna allowlista domen i klas pytań;
- jedna flaga powrotu do dotychczasowego retrievalu;
- automatyczny fallback SQL przy Qdrant offline;
- ręczny przegląd konfliktów high-risk;
- brak usuwania legacy przed zakończeniem okresu obserwacji.

### Etap F — przełączenie planera

Dopiero po etapach B–E:

- planner i jawny recall korzystają z jednego `AdaptiveHybridRetriever`;
- legacy payload content przestaje być źródłem treści;
- Qdrant zwraca wyłącznie referencje i ranking;
- SQL hydratuje finalny `ContextPackV1`;
- stare ścieżki są usuwane w osobnym PR po pomiarze rollbacku.

## Testy, które użytkownik może wykonać dziś

### 1. Mechaniczna regresja

```powershell
cd C:\Users\marci\VoiceLoop\listener
.\.venv\Scripts\python.exe -m ruff check voiceloop ..\tests ..\vectorscope `
  ..\scripts\voice_capture_server.py `
  ..\scripts\holding-commands\server.py `
  ..\scripts\calibration-phrases\server.py
.\.venv\Scripts\python.exe -m pytest -c pyproject.toml -q
```

### 2. Kopia danych przed indeksem

Zatrzymaj VoiceLoop i Qdrant albo wykonaj ich poprawny snapshot. Następnie:

```powershell
Copy-Item ..\data\voiceloop.db `
  "..\data\voiceloop.db.backup-$((Get-Date).ToString('yyyyMMdd-HHmmss'))"
```

Kopia samego SQLite nie jest snapshotem Qdrant.

### 3. Dry-run indeksu

```powershell
cd C:\Users\marci\VoiceLoop\listener
.\.venv\Scripts\python.exe -m voiceloop.corpus index-context-episodes `
  --database ..\data\voiceloop.db
```

Sprawdź `considered`, `would_index`, `not_requested` i nazwę kolekcji.
Dry-run nie łączy się z Qdrant.

### 4. Jawne indeksowanie

Wykonuj tylko przy lokalnym endpointcie embeddingów i lokalnym Qdrant:

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus index-context-episodes `
  --database ..\data\voiceloop.db --apply
```

Powtórz komendę. Drugi przebieg powinien wykazać `already_indexed`, a nie
ponowne embeddingowanie niezmienionych rekordów.

### 5. Status rewizji indeksu

Po uruchomieniu serwera pobierz token z `/api/v1/session` i sprawdź:

```powershell
$session = Invoke-RestMethod http://127.0.0.1:8765/api/v1/session
$headers = @{ "X-VoiceLoop-Token" = $session.token }
Invoke-RestMethod `
  -Headers $headers `
  http://127.0.0.1:8765/api/v1/context/dual-track/status |
  ConvertTo-Json -Depth 8
```

Przed uznaniem indeksu za zdrowy:

- `coverage.consistent` powinno być `true`;
- `coverage.pending`, `coverage.failed`, `coverage.delete_pending` i
  `coverage.untracked` powinny wynosić `0`;
- `index_state.outbox_ready` powinno wynosić `0`.

### 6. Prywatny gold set

Plik trzymaj poza repo. Minimalny rekord JSONL:

```json
{"schema_version":1,"example_id":"windows-001","query":"Co ustaliliśmy wczoraj o pliku konfiguracyjnym?","expected_source_ids":["<zatwierdzone-source-id>"],"expected_axes":["decision"],"observed_at":"2026-11-16T20:00:00+01:00","category":"windows","risk":"medium","expected_time_bounded":true}
```

Przygotuj osobno:

- 10–20 factual Windows;
- 10–20 exploratory Windows;
- 10–20 RPG canon;
- 10–20 RPG creative;
- co najmniej 20% negatywów;
- przykłady z 2–3 poprawnymi źródłami;
- konflikty i parafrazy bez wspólnych słów.

Nie commituj prywatnych nazw, transkryptów ani ścieżek.

### 7. Porównanie domen

```powershell
.\.venv\Scripts\python.exe -m voiceloop.corpus `
  report-adaptive-context-retrieval `
  --gold C:\PRIVATE\voiceloop-windows-gold.jsonl `
  --database ..\data\voiceloop.db `
  --domain windows

.\.venv\Scripts\python.exe -m voiceloop.corpus `
  report-adaptive-context-retrieval `
  --gold C:\PRIVATE\voiceloop-rpg-gold.jsonl `
  --database ..\data\voiceloop.db `
  --domain rpg
```

Porównaj source Recall@k, HitRate@k, Precision@k, nDCG@k, false-positive rate,
abstention accuracy, `vector_expansion_rate`, klasy pytań i powody stopu.

### 8. Próby awarii

Na kopii danych:

1. Wyłącz Qdrant i potwierdź, że SQLite/FTS nadal odpowiada, a indeks nie jest
   raportowany jako pusty/synchronizowany.
2. Przywróć Qdrant i ponów indexer; status `failed` powinien przejść w
   `applied`.
3. Zmień jawnie lokalny model embeddingowy i uruchom indexer; fingerprint
   powinien wymusić reindeksację.
4. Powtórz indexer bez zmian; liczba nowych zapisów powinna wynosić zero.
5. Sprawdź pusty wynik z jawnym czasem — nie może spaść do unbounded legacy
   recall.

Nie testuj retencji przez ręczne kasowanie produkcyjnych wierszy.

## Mapa kodu

- `context/policy.py` — wersjonowana polityka dynamiczna;
- `context/adaptive_retrieval.py` — FTS + adaptacyjna kaskada + fuzja;
- `context/retrieval.py` — czas, SQL gate, fale i weryfikacja;
- `context/vectorization.py` — jawny indexer i fingerprint;
- `memory.py` — kanon, outbox, desired/applied state;
- `qdrant_memory.py` — named vectors i ranking;
- `context/evaluation.py` — metryki jakości;
- `corpus/cli.py` — jawne raporty i operacje;
- `tests/test_context_index_coupling.py` — awarie i rewizje indeksu;
- `tests/test_adaptive_retrieval.py` — domeny i adaptacja;
- `tests/test_context_evaluation_metrics.py` — semantyka metryk.

## Decyzje wymagające użytkownika

Przed kolejnym etapem trzeba zatwierdzić:

1. skąd pochodzi zaufana domena i co zrobić przy niejednoznaczności;
2. czy kanoniczną jednostką odpowiedzi jest event, episode czy jawna memory;
3. budżet P95, CPU/RAM i maksymalną liczbę fal;
4. kryteria non-inferiority osobno dla Windows, RPG canon i RPG creative;
5. retencję: tombstone kontra hard delete dla poszczególnych źródeł;
6. lokalny model embeddingowy i moment zmiany generacji kolekcji;
7. operatora canary/rollbacku.

Bez tych decyzji kod pozostaje shadow i nie powinien sterować odpowiedzią live.
