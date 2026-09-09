# VoiceAttack Profile v2 PRO

VoiceAttack pełni rolę niezawodnego przycisku głosowego i warstwy awaryjnej.
Profil ma 670 jawnych polskich wariantów fraz: formy naturalne, krótkie skróty,
odmiany i typowe warianty bez polskich znaków. Swobodną wypowiedź po komendzie
**„Asystent”** albo **„Kursor”** nadal rozpoznaje Deepgram, nie wildcard VoiceAttack.

Routing komend jest automatyczny:

- przy wyłączonym Deepgramie znane komendy lecą bezpośrednio przez `CommandId`,
- przy aktywnym Deepgramie znane komendy VoiceAttack są mapowane na naturalny
  tekst i idą ścieżką tekstową, więc nie rozrywają sesji Venice,
- wypowiedzi spoza 656 wariantów obsługuje `Asystent` albo ciągły nasłuch.

Gotowy profil:

`C:\Users\marci\VoiceLoop\voiceattack\VoiceLoop-v2.vap`

Profil nazywa się **VoiceLoop v2 PRO**, zawiera 34 komendy i korzysta wyłącznie z
lokalnego API VoiceLoop na `127.0.0.1:8765`.

Generator odrzuca zduplikowane frazy pomiędzy komendami i komendę bez
istniejącego skryptu `.vbs`. Dzięki temu rozbudowanie słownika nie tworzy
niejednoznacznego routingu. Pełne 670 wariantów jest w
`scripts\build-voiceattack-profile.py`; niżej są najważniejsze przykłady.

Profil ustawia dla komend VoiceLoop próg rozpoznania `65`. Jeśli w logu
VoiceAttack nadal pojawia się komunikat `rejected with confidence level 73/75`,
to odrzuca globalne ustawienie VoiceAttack. Wtedy w opcjach VoiceAttack obniż
globalny minimalny próg rozpoznania z `75` do około `65-70`.

## Jak dokładnie działa „Asystent”

Rozmowa jest dwustopniowa:

1. Powiedz **„Asystent”** albo **„Hej asystent”**.
2. VoiceAttack uruchomi `scripts\va\assistant.vbs`.
3. Skrypt wywoła lokalny endpoint `POST /api/v1/listening/once`.
4. VoiceLoop powie **„Słucham”**.
5. Dopiero teraz wypowiedz pełne polecenie, na przykład:
   **„Podsumuj, czym zajmowałem się rano i przygotuj krótką odpowiedź”**.
6. Deepgram rozpozna jedną polską wypowiedź i automatycznie zamknie nasłuch.
   Jeśli nic nie powiesz, tryb wyłączy się po 30 sekundach.
7. Rdzeń zapisze żądanie, sprawdzi n8n i router deterministyczny, pobierze lokalny
   kontekst z pamięci wektorowej, a dopiero dla wolnego języka użyje Venice.
8. Venice zwróci wyłącznie ustrukturyzowany plan z dozwolonymi `action_id`.
   Gdy Venice jest niedostępny, plan może przejąć lokalny Qwen.
9. Executor sprawdzi ryzyko i ewentualne potwierdzenie. VoiceLoop wypowie
   odpowiedź lub końcowy wynik akcji.

Schemat:

```text
„Asystent”
  → VoiceAttack (pewna, stała fraza)
  → lokalne /listening/once
  → „Słucham”
  → następna wypowiedź
  → Deepgram STT
  → n8n → router deterministyczny → lokalny retrieval
  → Venice primary / Qwen fallback
  → allowlista akcji + potwierdzenie
  → executor → odpowiedź głosowa
```

Dwustopniowy tryb jest celowy. Polski Deepgram znacznie lepiej rozpoznaje dowolne
zdania niż wildcard systemowego silnika VoiceAttack. Stałe polecenia, takie jak
„otwórz kalendarz”, wypowiadaj bez poprzedzania ich słowem „Asystent”.

## Paleta sterowania kursorem i aktywnym folderem

„Kursor” działa identycznie jak „Asystent”, ale z innym pytaniem i pod kątem
bezpiecznego sterowania kursorem myszy oraz aktywnym folderem:

1. Powiedz **„Kursor”** (albo parafrazę: „Sterowanie kursorem”, „Ruch kursora”,
   „Mysz”, „Wskaźnik”).
2. VoiceAttack uruchomi `scripts\va\cursor.vbs`, który wywoła
   `POST /api/v1/listening/once?mode=cursor` — wyłącznie lokalne API VoiceLoop.
3. VoiceLoop powie: **„Gdzie przesunąć kursor lub co zrobić w aktywnym folderze?”**
4. Wypowiedz jedną z komend poniżej. Deepgram rozpozna ją i przekaże jako pełny
   tekst do bezpiecznego routera deterministycznego VoiceLoop.

**„Aktywny folder”** oznacza wyłącznie okno Eksploratora, które w danym momencie
znajduje się *pod kursorem myszy* — nigdy pulpit i nigdy inne okno wybrane
automatycznie. Jeśli pod kursorem nie ma okna Eksploratora, VoiceLoop odpowie
błędem i nic nie zaznaczy.

### Komendy palety

| Wypowiedź | Akcja | Ryzyko / potwierdzenie |
|---|---|---|
| `Najedź na ATAK` | `hover_shell_item` — przesuwa kursor na widoczną ikonę/element; nie klika. | niskie |
| `Przesuń kursor na Mortal Shell` | `hover_shell_item` | niskie |
| `Wróć kursorem` | `cursor_return` — przywraca pozycję kursora sprzed ostatniego przesunięcia. | niskie |
| `Kursor na środek` | `cursor_center` — przesuwa kursor na środek monitora, na którym aktualnie jest. | niskie |
| `Zaznacz folder NAZWA` | `select_shell_folder` — zaznacza jednoznaczny folder w aktywnym folderze przez UIA `SelectionItem`. | niskie |
| `Zaznacz plik NAZWA` | `select_shell_file` — jak wyżej, dla pliku. | niskie |
| `Zaznacz wszystkie PDF` | `select_shell_items_by_extension` — zaznacza wszystkie pliki danego rozszerzenia (wielokrotny UIA `SelectionItem`, bez przeciągania). | niskie |
| `Zaznacz wszystkie na literę A` | `select_shell_items_by_letter` | niskie |
| `Pierwszy` / `Drugi` / `Trzeci` | `select_listed_candidate` — wybiera jednego z 2-3 kandydatów zwróconych, gdy nazwa była niejednoznaczna. | niskie |
| `Otwórz Mortal Shell` / `Uruchom A Way Out` | `open_shell_item` — UIA Invoke albo Select+Enter, **zawsze wymaga potwierdzenia**. | **średnie, potwierdzenie** |
| `Przesuń okno na lewą połowę` / `prawą połowę` / `lewą jedną trzecią` / `środkową jedną trzecią` / `prawą jedną trzecią` / `lewą górną ćwiartkę` / `lewą dolną ćwiartkę` / `prawą górną ćwiartkę` / `prawą dolną ćwiartkę` | `snap_window_layout` — liczy prostokąt względem obszaru roboczego monitora aktywnego okna; nie używa menu Snap Layout. | niskie |
| `Stop` | zatrzymuje nasłuch, TTS i bieżącą akcję jak zawsze. | — |

### Niejednoznaczne nazwy

Router najpierw szuka dopasowania dokładnego (exact match). Gdy nie ma
dokładnego trafienia, dopasowanie rozmyte wymaga wysokiego podobieństwa oraz
wystarczającego marginesu nad drugim kandydatem, inaczej VoiceLoop nie wykonuje
żadnego ruchu. Przy pewnym wyniku wybierany jest jeden cel. Przy niepewności
VoiceLoop wymienia od 2 do 3 najlepszych kandydatów **z aktywnego, zweryfikowanego
okna** i nic nie otwiera ani nie zaznacza automatycznie — powiedz „Pierwszy”,
„Drugi” albo „Trzeci”, żeby wybrać.

### Przykłady

- **„Kursor”** → **„Najedź na ATAK”** — kursor przesuwa się na ikonę/element „ATAK”.
- **„Kursor”** → **„Zaznacz wszystkie PDF”** — zaznacza wszystkie pliki PDF w
  aktywnym folderze pod kursorem.
- **„Kursor”** → **„Przesuń okno na prawą jedną trzecią”** — aktywne okno zajmuje
  prawą jedną trzecią obszaru roboczego swojego monitora.
- **„Kursor”** → **„Otwórz Mortal Shell”** → **„Potwierdź”** — cel jest wiązany od
  razu, ale otwarcie następuje dopiero po ponownej walidacji UIA po słowie
  „Potwierdź”.

### Bezpieczeństwo palety

- Zero AutoHotkey, zero współrzędnych podawanych przez model językowy i zero
  ślepych kliknięć — kursor przesuwa wyłącznie zweryfikowany, jednoznaczny cel
  UIA, a zaznaczanie odbywa się przez UIA `SelectionItem` (`Select` /
  `AddToSelection`), nigdy przez przeciąganie.
- „Aktywny folder” nigdy nie spada automatycznie na pulpit ani na inne okno —
  gdy pod kursorem nie jest Eksplorator, akcja kończy się błędem po polsku.
- „Otwórz”/„Uruchom” to ryzyko średnie i zawsze wymaga „Potwierdź”. Cel jest
  wiązany od razu (przed potwierdzeniem), a po potwierdzeniu VoiceLoop
  ponownie waliduje ten sam element przez UIA, zanim go otworzy (UIA Invoke
  albo Select+Enter — nigdy podwójne kliknięcie).
- Układy okien liczą się względem obszaru roboczego monitora aktywnego okna,
  sprawdzają poprawność uchwytu okna i blokują pulpit, pasek zadań oraz inne
  chronione klasy systemowe. Nie korzystają z menu Snap Layout systemu Windows.

## Pakiet komend (34) z parafrazami

| Komenda główna | Przykładowe parafrazy | Działanie |
|---|---|---|
| `Asystent` | `Hej asystent`, `Słuchaj asystencie`, `Mam polecenie`, skrót: `Asys` | Otwiera jednorazowy nasłuch dowolnego polecenia. |
| `Kursor` | `Sterowanie kursorem`, `Ruch kursora`, `Mysz`, `Wskaźnik` | Otwiera jednorazowy nasłuch dla palety sterowania kursorem i aktywnym folderem. |
| `Zapisz notatkę` | `Nowa notatka`, `Utwórz notatkę`, `Zanotuj to`, skrót: `Notka` | Pyta o treść i zapisuje ją kontrolowaną akcją UI.Vision. |
| `Zapamiętaj` | `Zapamiętaj to`, `Pamiętaj to`, `Zapisz to w pamięci`, skrót: `Pamiętaj` | Pyta o fakt i przygotowuje lokalny zapis wymagający zgody. |
| `Co robiłem ostatnio` | `Co się działo na ekranie`, `Pokaż historię ekranu`, `Podsumuj aktywność`, skrót: `Aktywność` | Czyta lokalne podsumowanie aktywności Screenpipe. |
| `Opisz aktywne okno` | `Co mam otwarte`, `Jakie okno jest aktywne`, `Na jakim oknie jestem`, skrót: `Aktywne okno` | Lokalnie odczytuje aktywne okno i program. |
| `Sprawdź pole tekstowe` | `Gdzie teraz piszę`, `Czy to pasek adresu`, `Gdzie trafi tekst` | Sprawdza, czy wpisywanie trafi do właściwego pola, a nie np. paska adresu. |
| `Zminimalizuj okno` | `Schowaj okno`, `Ukryj okno`, `Zwiń aktywne okno`, skrót: `Zwiń` | Minimalizuje aktualnie aktywne okno. |
| `Zminimalizuj okno pod kursorem` | `Schowaj okno pod kursorem`, `Zwiń okno pod myszką` | Minimalizuje okno wskazywane kursorem. |
| `Zminimalizuj wszystkie` | `Pokaż pulpit`, `Minimalizuj wszystko`, `Zwiń wszystkie okna`, skrót: `Pulpit` | Minimalizuje wszystkie okna i pokazuje pulpit. |
| `Wyłącz aplikację pod kursorem` | `Zamknij wskazane okno`, `Zamknij program który wskazuję`, `Zamknij to okno` | Po potwierdzeniu wysyła `WM_CLOSE`; nie zabija procesu i pozostawia pytanie o zapis. |
| `Kopiuj zaznaczony tekst` | `Skopiuj zaznaczony tekst`, `Kopiuj zaznaczony fragment`, skrót: `Kopiuj zaznaczenie` | Kopiuje aktualnie zaznaczony tekst do schowka. |
| `Kopiuj tekst pod kursorem` | `Kopiuj spod kursora`, `Skopiuj wskazany tekst`, `Wskazany tekst do schowka` | Kopiuje najbliższy tekst udostępniony przez UI Automation. |
| `Kopiuj email pod kursorem` | `Skopiuj mail pod myszką`, `Adres e-mail do schowka`, `Skopiuj mail który wskazuję` | Kopiuje jeden jednoznaczny adres e-mail z elementu pod kursorem. |
| `Kopiuj numer pod kursorem` | `Skopiuj numer pod kursorem`, `Kopiuj liczbę pod kursorem`, `Skopiuj telefon pod kursorem`, skrót: `Kopiuj numer` | Wykrywa i kopiuje numer pod kursorem. |
| `Kopiuj całe zdanie pod kursorem` | `Skopiuj całe zdanie pod kursorem`, `Skopiuj zdanie pod myszką`, skrót: `Kopiuj zdanie` | Wykrywa i kopiuje całe zdanie pod kursorem. |
| `Zaznacz zdanie pod kursorem` | `Zaznacz całe zdanie`, `Wybierz wskazane zdanie`, `Podświetl zdanie pod kursorem` | Zaznacza zdanie przez tekstowy zakres UI Automation. |
| `Zaznacz akapit pod kursorem` | `Zaznacz cały akapit`, `Wybierz wskazany akapit`, `Zaznacz paragraf` | Zaznacza akapit przez tekstowy zakres UI Automation. |
| `Włącz nasłuch` | `Zacznij nasłuch`, `Start nasłuchu`, `Słuchaj ciągle`, skrót: `Nasłuch on` | Uruchamia ciągły nasłuch Deepgram. |
| `Wyłącz nasłuch` | `Zatrzymaj nasłuch`, `Stop nasłuchu`, `Przestań słuchać`, skrót: `Nasłuch off` | Wyłącza tylko Deepgram. |
| `Status Voice Loop` | `Czy działasz`, `Jaki jest status`, `Podaj status systemu`, skrót: `Status` | Czyta stan rdzenia, modelu i nasłuchu. |
| `Potwierdź` | `Tak potwierdzam`, `Wykonaj to`, `Zatwierdź`, skrót: `Potwierdź` | Potwierdza najnowszą akcję oczekującą na zgodę. |
| `Anuluj zadanie` | `Nie rób tego`, `Przerwij to`, `Odrzuć polecenie`, skrót: `Anuluj` | Odrzuca najnowszą akcję oczekującą na zgodę. |
| `Test pętli` | `Test głosu`, `Test VoiceLoop`, `Sprawdź głos`, skrót: `Test` | Sprawdza lokalną pętlę API i TTS. |
| `Otwórz kalendarz` | `Uruchom kalendarz`, `Pokaż kalendarz`, `Włącz kalendarz`, skrót: `Kalendarz` | Uruchamia dozwoloną akcję kalendarza. |
| `Otwórz przeglądarkę` | `Uruchom przeglądarkę`, `Otwórz internet`, `Odpal przeglądarkę`, skrót: `Przeglądarka` | Uruchamia dozwoloną akcję przeglądarki. |
| `Otwórz czat` | `Open chat`, `Otwórz ChatGPT`, `Nowy chat GPT`, skrót: `ChatGPT` | Uruchamia dozwoloną akcję czatu. |
| `Otwórz GPT` | `Open GPT`, `Otwórz chat GPT`, `Uruchom GPT` | Otwiera osobny czat GPT. |
| `Otwórz Gemini` | `Open Gemini`, `Czat Gemini`, `Uruchom Gemini` | Otwiera osobny czat Gemini. |
| `Wyszukaj w internecie` | `Sprawdź w necie`, `Szukaj online` | Uruchamia szybkie wyszukiwanie internetowe (potem doprecyzuj zapytanie). |
| `Zapamiętaj ostatnie źródło` | `Zapisz ostatni link`, `Zachowaj wynik wyszukiwania`, `Zapisz to źródło` | Przygotowuje zapis ostatniego wyniku internetowego i wymaga potwierdzenia. |
| `Co potrafisz` | `Jakie masz możliwości`, `Lista komend` | Czyta możliwości VoiceAttack i dodatkowe akcje VoiceLoop. |
| `Zmień nazwę` | `Zmień nazwę pod kursorem`, `Przemianuj to` | Przygotowuje zmianę nazwy elementu pod kursorem i wymaga potwierdzenia. |
| `Stop teraz` | `Przerwij wszystko`, `Awaryjnie stop`, `Panic stop`, skrót: `Stop` | Natychmiast zatrzymuje nasłuch, TTS, kolejkę i bieżącą akcję. |

## Bezpieczny audyt akcji

Audyt nie uruchamia żadnej komendy ani akcji ekranowej. Porównuje generator,
profil `.vap`, wrappery `.vbs` oraz żywy katalog allowlisty VoiceLoop:

```powershell
.\listener\.venv\Scripts\python.exe .\scripts\check_voiceattack_actions.py
```

Raport jest zapisywany w `logs\voiceattack-action-audit.json`. Tryb bez
uruchomionego listenera:

```powershell
.\listener\.venv\Scripts\python.exe .\scripts\check_voiceattack_actions.py --offline
```

### Ręczne uruchomienie skryptów minimalizacji

- `Uruchom: C:\Users\marci\VoiceLoop\scripts\va\minimize-window.vbs`
- `Uruchom: C:\Users\marci\VoiceLoop\scripts\va\minimize-all.vbs`

## Notatka i pamięć — różnica

### „Zapisz notatkę”

VoiceLoop odpowiada „Co zapisać w notatce?”, słucha jednej wypowiedzi, a następnie
tworzy notatkę. Treść nie musi przechodzić przez Venice, ponieważ lokalny router
rozpoznaje prefiks i buduje akcję `create_note`.

### „Zapamiętaj”

VoiceLoop odpowiada „Co mam zapamiętać?”, słucha faktu i tworzy akcję `remember`.
Nic nie zostanie zapisane od razu. Asystent poprosi o decyzję:

- **„Potwierdź”** — zapis do lokalnej pamięci;
- **„Anuluj zadanie”** — odrzucenie.

Zgoda wygasa po 5 minutach. To zabezpiecza pamięć przed przypadkowym zapisem
źle rozpoznanego zdania oraz przed wykonaniem starego polecenia po restarcie.

## Tryb jednorazowy a ciągły

- **„Asystent”**, **„Zapisz notatkę”** i **„Zapamiętaj”** otwierają mikrofon
  Deepgram tylko na jedną wypowiedź.
- **„Włącz nasłuch”** pozostawia Deepgram aktywny i każde zakończone zdanie
  przesyła jako nowe polecenie.
- **„Wyłącz nasłuch”** zatrzymuje tylko Deepgram.
- **„Stop teraz”** jest panic buttonem i zatrzymuje także akcje oraz TTS.

## Instalacja profilu

1. Uruchom cały stos przez `scripts\start-all.ps1`.
2. Otwórz VoiceAttack.
3. Wybierz **More Actions → Import Profile**.
4. Wskaż `C:\Users\marci\VoiceLoop\voiceattack\VoiceLoop-v2.vap`.
5. Wybierz na liście profil **VoiceLoop v2 PRO**.
6. Upewnij się, że VoiceAttack słucha i ma ustawiony właściwy mikrofon.
7. Powiedz **„Test pętli”**, a następnie **„Asystent”**.

Stare profile `VoiceLoop` i `VoiceLoop v2` mogą pozostać jako archiwum, ale
aktywny powinien być tylko `VoiceLoop v2 PRO`.

### Częsty błąd: komendy „wsiąkają” w inny profil

Objaw: „Kursor” bywa losowo rozpoznawany jako coś zupełnie innego, mimo
niskiego szumu. W logu VoiceAttack widać na przemian odrzucenia z progiem
`.../65` (to są komendy VoiceLoop — profil ustawia im próg 65) i z progiem
`.../75` lub innym (to są **cudze, stare komendy** z tego samego profilu).
Potwierdzony w nagraniu przypadek: stara komenda z frazą „otwórz cursor” (próg
75) uruchamiająca edytor `Cursor.exe`, obok naszej komendy „kursor” (próg 65,
`cursor.vbs`) — silnik czasem wybiera tę starą zamiast naszej.

Przyczyna: w **More Actions** VoiceAttack ma dwie różne opcje — **Import
Profile** (tworzy/nadpisuje osobny profil) i **Import Commands** (dogrywa
komendy do profilu, który jest akurat aktywny). Użycie tej drugiej opcji, albo
zgoda na „scalenie” zamiast „nadpisania” przy imporcie, miesza komendy
VoiceLoop ze starymi, niepowiązanymi komendami w jednym profilu (np. w
domyślnym `My Profile`).

Naprawa:
1. Sprawdź, który profil jest aktywny (dropdown `Profile` u góry okna VoiceAttack).
2. Jeśli widzisz w nim komendy spoza kategorii „VoiceLoop v2 PRO” (np. „otwórz
   cursor”, „hello”) — usuń ten profil całkowicie (**Delete Profile**), żeby
   nie dało się go przypadkiem znowu aktywować.
3. Zrób od nowa czysty **Import Profile** ze świeżo przebudowanego
   `VoiceLoop-v2.vap` i przy imporcie wybierz nadpisanie, nie scalenie.
4. Ustaw jako aktywny wyłącznie **VoiceLoop v2 PRO** i sprawdź w edytorze
   komend, że kategoria ma dokładnie 34 pozycje (tyle generuje skrypt).

## Test przyjęcia

1. **„Test pętli”** → „Pętla VoiceLoop działa”.
2. **„Status Voice Loop”** → krótki komunikat o modelu i Deepgramie.
3. **„Asystent”** → „Słucham” → powiedz „Jaki model jest teraz główny?”.
4. **„Opisz aktywne okno”** → asystent czyta rzeczywisty tytuł okna.
5. Najedź na adres → **„Skopiuj mail pod myszką”** → adres trafia do schowka.
6. Najedź na tekst → **„Zaznacz zdanie pod kursorem”** → zaznacza się zdanie.
7. Najedź na testowe okno → **„Wyłącz aplikację pod kursorem”** → dopiero
   **„Potwierdź”** wysyła zamknięcie.
8. **„Zapamiętaj”** → podaj fakt → **„Potwierdź”** → „Zapamiętano”.
9. **„Kursor”** → **„Najedź na ATAK”** → kursor przesuwa się na wskazany element.
10. Wskaż kursorem okno Eksploratora → **„Kursor”** → **„Zaznacz wszystkie PDF”**
    → zaznaczają się wszystkie pliki PDF w tym oknie.
11. **„Kursor”** → **„Otwórz Mortal Shell”** → dopiero **„Potwierdź”** otwiera element.
12. **„Stop teraz”** → aktywne zadania i głos zostają przerwane.

## Bezpieczeństwo

- Skrypty `.vbs` nie wykonują treści wypowiedzi jako PowerShell ani CMD.
- Wszystkie żądania wymagają lokalnego tokenu `X-VoiceLoop-Token`.
- STOP idzie bezpośrednio do rdzenia i omija n8n oraz modele.
- Screenpipe, aktywne okno i Qdrant są przetwarzane lokalnie.
- Do Venice trafia tekst wolnego polecenia i wybrany kontekst, nie dowolny plik.
- Model może wskazać tylko akcje z allowlisty.
- Akcje wysokiego ryzyka oraz zapis pamięci wymagają potwierdzenia.
- Zamknięcie okna pod kursorem wymaga potwierdzenia, blokuje pulpit i pasek zadań
  oraz używa `WM_CLOSE` zamiast zakończenia procesu. Nie przesuwaj kursora na
  inne okno przed powiedzeniem „Potwierdź”.
- Zaznaczanie zdania i akapitu nie używa awaryjnych kliknięć pikselowych. Jeśli
  aplikacja nie udostępnia UI Automation TextPattern, akcja kończy się błędem.

Profil można odtworzyć po zmianie ścieżki projektu:

```powershell
.\listener\.venv\Scripts\python.exe .\scripts\build-voiceattack-profile.py
```
