# Kaskada 2+2+1 i granica bezpieczeństwa

Stan na 19 października 2026. Źródłem prawdy pozostaje
`listener/voiceloop/`. Ten dokument opisuje pierwotny miernik shadow oraz
granice wspólne z izolowanym prototypem dual-track. Pełny kontrakt prototypu:
[`DUAL_TRACK_CASCADE_PROTOTYPE.md`](DUAL_TRACK_CASCADE_PROTOTYPE.md).

## Co jest wdrożone

Pamięć VoiceLoop ma sześć wymiarów. Czas nie jest wektorem. Pięć pozostałych
wymiarów to nazwane wektory w Qdrant.

| Wymiar dla człowieka | Identyfikator w kodzie | Gdzie żyje |
| --- | --- | --- |
| Czas | `event_time` | SQLite, czas UTC |
| Treść | `semantic` | Qdrant |
| Temat | `topic` | Qdrant |
| Cel | `intent` | Qdrant |
| Ustalenie | `decision` | Qdrant |
| Relacja | `person_context` | Qdrant |

Nazwy w lewej kolumnie są propozycją do zatwierdzenia. Kod nadal używa
identyfikatorów z prawej kolumny. Żaden ekran nie pokazuje jeszcze lewej
kolumny jako kanonicznych etykiet.

Ścieżka, która dziś buduje odpowiedź, jest bez zmian: czas i FTS w SQLite,
potem do pięciu zapytań wektorowych równolegle, potem Reciprocal Rank Fusion.
Kaskada tego nie zastępuje.

Obok tej ścieżki działa pomiar w tle, domyślnie wyłączony:

`CONTEXT_CASCADE_SHADOW_ENABLED=false`

Po włączeniu miernik układa osie falami. Pierwsza fala bierze Treść i jedną
oś wskazaną przez klasę zapytania. Druga fala bierze niewykorzystane osie.
Łącznie może zapytać co najwyżej pięć osi, więc przy pięciu osiach układ
2+2+1 daje pięć zapytań, nie sześć. Brak szóstej osi „kiedy”: czas jest już
rozstrzygnięty w SQL, zanim poleci pierwsze zapytanie wektorowe.

Pierwotny miernik zatrzymuje się według liczby zaakceptowanych pozycji. Nie
jest to gwarancja pokrycia dwóch osi ani kompletności decyzji. Prototyp
dual-track dodaje obok niego porównanie z RRF5 i deterministyczny werdykt,
ale nadal nie wybiera wyniku do odpowiedzi. Żaden z tych torów nie zmienia
`ContextPack` i nie wykonuje akcji.

## Bezpieczeństwo

Ta sekcja jest osobna, bo kaskada i twarde granice to różne warstwy. Kaskada
może być wyłączona. Granice poniżej działają także wtedy.

### 1. Pętla zwrotna HTTP

Serwer odrzuca żądanie, zanim dojdzie do trasy, gdy nagłówek `Host` nie jest
dokładnie `127.0.0.1`, `::1` albo `localhost`. Nagłówek `Origin`, jeśli jest,
musi wskazywać ten sam host i port. Dzięki temu przeglądarka z innej strony
oraz żądanie z przebindowanym DNS nie dostają tokenu sesji.

`GET /api/v1/session` oddaje token tylko klientowi, którego adresem połączenia
jest pętla zwrotna. Endpointy prywatne wymagają nagłówka
`X-VoiceLoop-Token`. Porównanie tokenu jest stałoczasowe.

Plik tokenu ma właściciela bieżącego użytkownika. Na Windowsie DACL jest
chroniony i daje dostęp tylko temu właścicielowi. Na POSIX uprawnienia pliku
to `0600`.

### 2. Wyszukiwanie sieci nie schodzi do sieci lokalnej

Gdy włączone jest opcjonalne wyszukiwanie w sieci, każdy adres jest
rozwiązywany przed połączeniem. Odrzucane są pętla zwrotna, sieci prywatne,
łącze lokalne, multicast i adres metadanych chmury `169.254.169.254`.
Przekierowanie jest sprawdzane tak samo, z limitem przeskoków. Treść zapytania
nie trafia do telemetrii tego kanału.

### 3. Pliki osi czasu nie wychodzą poza swój katalog

Zapis dokumentu sprawdza, że ścieżka po rozwiązaniu symlinków zostaje wewnątrz
katalogu osi czasu. Samo `is_relative_to` na ścieżce nierozwiązanej nie
wystarcza, więc kod rozwiązuje cel i sprawdza ponownie.

### 4. Retencja nie kasuje starych danych wstecz

Nowe nagrania spotkań dostają termin 7 dni. Nowe transkrypty poleceń dostają
termin 30 dni. Liczy się czas serwera w chwili zapisu, nie data przysyłana
przez klienta. Rekord bez terminu nie jest traktowany jako przeterminowany.
Migracja starych rekordów działa wyłącznie jako podgląd: pokazuje, ile wierszy
dostałoby termin, i niczego nie usuwa. Próba wymuszenia kasującej migracji
jest w tej wersji odrzucana przy starcie.

### 5. Cień nie staje się odpowiedzią

Miernik kaskady startuje dopiero po zbudowaniu właściwego pakietu kontekstu.
Błąd, timeout i brak Qdrant nie zmieniają odpowiedzi. Identyfikator w zdarzeniu
`context.retrieval_cascade_shadow` jest skrótem SHA-256, nie surowym
`request_id`. Zdarzenie niesie liczniki, fale, powód stopu i czas. Nie niesie
treści zapytania ani treści wspomnień.

### 6. Zdanie złożone nie staje się faktem

Jedna wąska forma jest rozpoznawana obok odpowiedzi, nadal tylko przy
włączonym mierniku. Zdanie w rodzaju „a jeszcze miesiąc temu mówiliśmy, że
będzie wspaniale” rozdziela dwie role czasu:

- czas wypowiedzi: przybliżone okno 23–37 dni przed chwilą obserwacji,
  pewność 0.65, źródło `relative_text`;
- czas treści: przyszłość względem wypowiedzi, precyzja `unknown`,
  pewność 0.55, źródło `semantic_relation`.

Rola epistemiczna to `prediction`, gdy tekst mówi o tym, co będzie, albo
`opinion`, gdy tekst mówi o tym, co ktoś myśli. Inne zdania złożone miernik
zostawia nietknięte.

Taki wynik nie dostaje osi faktu. Ma lukę `compound_claim_requires_review`,
dopóki człowiek nie zatwierdzi atomowej tezy, której cytat naprawdę występuje
w zdaniu. Zapis lokalny trzyma role czasu i skrót pochodzenia. Surowe zdanie
zostaje w wierszu polecenia, do którego i tak należy.

Pytanie porównawcze, sprzeczność, kontekst decyzji albo skutek działania
podnoszą bramkę do `investigate`. To jest etykieta pomiaru. Nie uruchamia
akcji i nie dopisuje wydarzenia do osi czasu.

### Czego ta wersja nie zamyka

- Szerszy indeks pamięci, `context_timeline_v1`, nadal jest eksperymentem.
  Pierwotny miernik kaskady czyta zwykłą kolekcję pamięci. Prototyp dual-track
  czyta wyłącznie izolowaną kolekcję epizodów.
- Pięć osi nadal może być zapytanych naraz na ścieżce odpowiedzi.
- Trwałe werdykty dual-track są pomiarem algorytmu, nie werdyktem człowieka
  i nie stają się kanoniczną wartością.
- Formy czasu inne niż „miesiąc temu” plus predykcja albo opinia nie są
  rozpoznawane.
- Włączenie cienia na maszynie z Qdrant zwiększa liczbę zapytań obok
  odpowiedzi. Dlatego domyślnie jest wyłączony.

Testy tych granic są w `tests/test_api_privacy.py`, `tests/test_web_search.py`,
`tests/test_context_phase3.py`, `tests/test_memory.py`,
`tests/test_context_retrieval_integrity.py`, `tests/test_context_dual_track.py`
i `tests/test_assistant.py`.

## Praca człowieka

Poniższe punkty nie są w kodzie do „dokończenia przez model”. Każdy wymaga
Twojej decyzji. Dopóki punkt jest otwarty, odpowiadająca funkcja zostaje
wyłączona albo pozostaje tylko propozycją.

1. **Kalibracja punktów niepewności.** Wskaż, przy których sygnałach asystent
   ma się zatrzymać i zapytać, zamiast zgadywać. Proponowany pierwszy zestaw:
   brak czasu, konflikt dwóch czasów tego samego wydarzenia, oś poniżej progu,
   zdanie złożone bez zatwierdzonej tezy, akcja o wysokim skutku. Dla każdego
   punktu zapisz werdykt: kandydat, wartość kanoniczna albo odrzucenie.
   Kod nie ma jeszcze kolejki tych werdyktów. Bez tego punktu nie wolno
   przełączać kaskady na ścieżkę odpowiedzi.

2. **Nadanie nazw osi.** Zatwierdź albo zmień nazwy z tabeli na górze:
   Czas, Treść, Temat, Cel, Ustalenie, Relacja. Identyfikatory techniczne
   zostają. Do interfejsu wolno wziąć tylko nazwy, które zatwierdzisz.
   Nie dodajemy osi „kiedy”.

3. **Retencja danych zapisanych wcześniej.** Nowe rekordy już dostają termin.
   Stare rekordy bez terminu zostają. Zanim powstanie jakakolwiek kasująca
   migracja, przejrz raport podglądu: liczba nagrań, liczba transkryptów,
   najstarsza data. Ta wersja świadomie odmawia kasowania wstecz.

4. **Włączenie miernika.** Na własnej maszynie, gdy chcesz pomiar, ustaw
   `CONTEXT_DUAL_TRACK_ENABLED=true` dla izolowanego porównania. Zostaw
   `CONTEXT_CASCADE_SHADOW_QMAX=5` i
   `CONTEXT_CASCADE_SHADOW_WAVE_SIZE=2`. Miernik ma pisać tylko zredagowane
   zdarzenia, liczniki i content-free trace’y z TTL. Jeśli odpowiedź asystenta
   się zmieni, wyłącz flagę: to błąd, nie oczekiwany efekt.

5. **Prywatny zbiór wzorcowy.** Zanim kaskada w ogóle wejdzie w rozmowę
   o przełączeniu na odpowiedzi, oznacz własne pytania: trafienie, pudło,
   powód stopu, czy czas był ograniczony. Bez tego zbioru liczba zapytań
   nie jest dowodem jakości.

6. **Usługi lokalne.** Wektory działają tylko z embeddingiem na pętli zwrotnej
   i z Qdrant, który przyjmuje prywatne dane. Ty uruchamiasz te procesy
   i Ty decydujesz, czy ich treść może opuścić maszynę. Przy braku usługi
   zostaje SQLite i FTS. To jest poprawny stan, nie awaria do obejścia.

Kolejność jest wiążąca: najpierw 1 i 2, potem 3 jako osobna zgoda, potem 4
i 5 jako pomiar. Punkt 6 jest warunkiem pomiaru wektorowego, nie warunkiem
działania asystenta.
