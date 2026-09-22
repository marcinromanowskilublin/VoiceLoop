# Nawigacja między osiami pamięci — kierunek projektowania

**Status: uzgodniony kierunek, bez wdrożonego silnika nawigacji.**
**Data:** 22 września 2026. **Aktualny kod:** VoiceLoop 0.3.0, po `bf26abe`.

Docelowo pamięć ma mieć większy zestaw wyspecjalizowanych osi znaczeniowych:
więcej niż obecne pięć, ale nie setki. Główną wartością mają być dobrze
określone reguły przechodzenia między nimi. Liczba i nazwy dodatkowych osi
pozostają do uzgodnienia. Większy katalog nie oznacza odpytywania wszystkich
osi przy każdym pytaniu.

## Co istnieje dzisiaj

- Pięć osi pamięci: `semantic`, `topic`, `intent`, `decision`, `person_context`.
- Planer i domyślny recall tworzą pięć reprezentacji pytania i łączą rankingi.
- Opcjonalny time-first recall szuka w SQLite/FTS i Screenpipe, potem próbuje
  `semantic` oraz wybrane pytaniem osie rezerwowe. Zatrzymuje się głównie według
  liczby przyjętych pozycji.
- Kandydaci semantyczni tej ścieżki muszą mieć aktualny epizod i niezmienione,
  żywe źródła SQL. Ranking pozostaje oddzielony od `confidence`.
- Capabilities mają osobny indeks `semantic` / `intent` / `target_context`.
  Ten projekt dotyczy pamięci, nie zmiany indeksu doboru działań.

Aktualne kontrakty: [Context Timeline V1](CONTEXT_TIMELINE_V1.md) i
[architektura](ARCHITECTURE_CURRENT.md).

## Docelowa ścieżka do zaprojektowania

```text
pytanie i jego zakres
  -> klasa pytania oraz wymagania dowodowe
  -> SQL/FTS: czas, źródła, zatwierdzone encje i dokładne nazwy
  -> wybrane osie semantyczne, gdy pozostała konkretna luka
  -> potwierdzenie kandydatów w kanonicznym SQL
  -> następna oś / odpowiedź / doprecyzowanie / brak wystarczających dowodów
```

Planer i jawny recall powinny korzystać ze wspólnego wejścia do pamięci,
z zachowaniem źródeł i zakresu pytania. W zapisie kierunek to kanoniczny SQL
oraz kolejka indeksowania pochodnych wektorów. Wspólne wejście i kolejka nie
są jeszcze wdrożone. Obecne poprawki weryfikacji SQL są fundamentem, a nie
implementacją całego powyższego przepływu.

## Ranga pytania, trudność i wystarczające dowody

Wymagania dowodowe mają zależeć od rangi pytania i problemu. Trzeba osobno
określić:

| Wymiar | Pytanie projektowe | Wpływ na przyszłe wyszukiwanie |
|---|---|---|
| Waga i konsekwencje pomyłki | Co się stanie, jeżeli odpowiedź będzie błędna? | Wymagane źródła, stopień potwierdzenia, granica abstencji. |
| Trudność szukania | Jak niejednoznaczne, rozproszone lub niepełne są dane? | Wybór osi, kolejne przejścia i budżet. |

Łatwe do znalezienia twierdzenie może mieć poważne konsekwencje błędu;
trudne szukanie nie musi oznaczać ważnej decyzji. Nie należy zastępować tych
dwóch ocen jedną liczbą podobieństwa wektorowego.

AI proponuje klasy pytań i przykłady. Użytkownik ustala ich znaczenie oraz
oczekiwany poziom pewności na własnych przykładach. Dokładne progi, budżety,
liczba przykładów i nazwy klas nie są jeszcze zaakceptowaną specyfikacją.

## Co musi określać mapa osi

Dla każdej osi potrzebny jest opis znaczenia, rodzaju indeksowanych danych,
źródeł oraz przykładów pasujących i niepasujących pytań. Reguły nawigacji
muszą określać:

- warunek wejścia na oś i brak dowodu, który ma ona uzupełnić;
- warunek przejścia do następnej osi i maksymalny budżet;
- wymagane potwierdzenie tożsamości, wersji, czasu i pochodzenia w SQL;
- warunki odpowiedzi, zatrzymania, pytania doprecyzowującego i abstencji;
- zachowanie przy sprzecznych źródłach, braku danych i niedostępności usługi.

Warunki zatrzymania muszą być oceniane na oznaczonych przykładach. Sama liczba
trafień, wysoki RRF albo udany test techniczny nie potwierdzają, że materiał
wystarcza do odpowiedzi. Użytkownik zatwierdza wzorzec; późniejsze zapytania
mają poruszać się między osiami automatycznie.

## Podział pracy

| Odpowiedzialność | Zadanie | Warunek zakończenia |
|---|---|---|
| AI | Zaproponować słownik osi, ścieżki, symulacje, miary jakości i kosztu. Przygotować neutralne zestawy do oznaczenia. | Projekt możliwy do oceny na przykładach, z jawnymi założeniami i brakami. |
| Użytkownik | Ustalić prywatne znaczenia kategorii i nazw, potwierdzić rozmówców, powiązania osób/projektów oraz decyzje/propozycje. Wskazać oczekiwane odpowiedzi i źródła. | Potwierdzone albo jawnie nieznane etykiety konkretnego zestawu; propozycje AI pozostają oddzielone od potwierdzeń. |
| Wspólnie | Zatwierdzić mapę, klasy pytań, wymagania dowodowe i reguły przejść/zatrzymania. | Słownik osi z opisami, przykłady pytań i oczekiwanych źródeł/ścieżek, jawne reguły oraz kryteria oceny. |
| AI po zatwierdzeniu | Wdrożyć uzgodniony wzorzec i wykonać regresje oraz porównanie jakości. | Zmierzone zachowanie dla konkretnej wersji kodu, modelu i danych; brak automatycznego awansu do działania na żywo. |

Ten dokument nie zleca skanowania prywatnych treści ani samodzielnego
zatwierdzania ich przez AI. Liczba brakujących etykiet nie jest zgadywana.
Nagranie własnego głosu i odsłuch nadal wymagają osobistego udziału użytkownika;
nie są zastępowane kalibracją wyników wyszukiwania.
