# VoiceAttack Profile v2 PRO

VoiceAttack pełni rolę niezawodnego przycisku głosowego i warstwy awaryjnej.
Profil ma 656 jawnych polskich wariantów fraz: formy naturalne, krótkie skróty,
odmiany i typowe warianty bez polskich znaków. Swobodną wypowiedź po komendzie
**„Asystent”** nadal rozpoznaje Deepgram, nie wildcard VoiceAttack.

Routing komend jest automatyczny:

- przy wyłączonym Deepgramie znane komendy lecą bezpośrednio przez `CommandId`,
- przy aktywnym Deepgramie znane komendy VoiceAttack są mapowane na naturalny
  tekst i idą ścieżką tekstową, więc nie rozrywają sesji Venice,
- wypowiedzi spoza 656 wariantów obsługuje `Asystent` albo ciągły nasłuch.

Przed importem wygeneruj profil. Generator osadza aktualną ścieżkę sklonowanego
repozytorium w akcjach VoiceAttack:

```powershell
.\listener\.venv\Scripts\python.exe .\scripts\build-voiceattack-profile.py
```

Wynik zapisuje się w `voiceattack\VoiceLoop-v2.vap`. Profil nazywa się
**VoiceLoop v2 PRO**, zawiera 33 komendy i korzysta wyłącznie z
lokalnego API VoiceLoop na `127.0.0.1:8765`.

Generator odrzuca zduplikowane frazy pomiędzy komendami i komendę bez
istniejącego skryptu `.vbs`. Dzięki temu rozbudowanie słownika nie tworzy
niejednoznacznego routingu. Pełne 656 wariantów jest w
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

## Pakiet komend (30) z parafrazami

| Komenda główna | Przykładowe parafrazy | Działanie |
|---|---|---|
| `Asystent` | `Hej asystent`, `Słuchaj asystencie`, `Mam polecenie`, skrót: `Asys` | Otwiera jednorazowy nasłuch dowolnego polecenia. |
| `Zapisz notatkę` | `Nowa notatka`, `Utwórz notatkę`, `Zanotuj to`, skrót: `Notka` | Pyta o treść i zapisuje ją kontrolowaną akcją UI.Vision. |
| `Zapamiętaj` | `Zapamiętaj to`, `Pamiętaj to`, `Zapisz to w pamięci`, skrót: `Pamiętaj` | Pyta o fakt i przygotowuje lokalny zapis wymagający zgody. |
| `Co robiłem ostatnio` | `Co się działo na ekranie`, `Pokaż historię ekranu`, `Podsumuj aktywność`, skrót: `Aktywność` | Czyta lokalne podsumowanie aktywności Screenpipe. |
| `Opisz aktywne okno` | `Co mam otwarte`, `Jakie okno jest aktywne`, `Na jakim oknie jestem`, skrót: `Aktywne okno` | Lokalnie odczytuje aktywne okno i program. |
| `Sprawdź pole tekstowe` | `Gdzie teraz piszę`, `Czy to pasek adresu`, `Gdzie trafi tekst` | Sprawdza, czy wpisywanie trafi do właściwego pola, a nie np. paska adresu. |
| `Zminimalizuj okno` | `Schowaj okno`, `Ukryj okno`, `Zwiń aktywne okno`, skrót: `Zwiń` | Minimalizuje aktualnie aktywne okno. |
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
| `Stop teraz` | `Przerwij wszystko`, `Awaryjnie stop`, `Panic stop`, skrót: `Stop` | Natychmiast zatrzymuje nasłuch, TTS, kolejkę i bieżącą akcję. |

### Ręczne uruchomienie skryptów minimalizacji

- `Uruchom: <ścieżka-do-repozytorium>\scripts\va\minimize-window.vbs`
- `Uruchom: <ścieżka-do-repozytorium>\scripts\va\minimize-all.vbs`

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

1. Wygeneruj profil powyższą komendą.
2. Uruchom rdzeń przez `scripts\start-core.bat`; opcjonalne usługi włączaj tylko
   dla funkcji, które ich wymagają.
3. Otwórz VoiceAttack.
4. Wybierz **More Actions → Import Profile**.
5. Wskaż `voiceattack\VoiceLoop-v2.vap`.
6. Wybierz na liście profil **VoiceLoop v2 PRO**.
7. Upewnij się, że VoiceAttack słucha i ma ustawiony właściwy mikrofon.
8. Powiedz **„Test pętli”**, a następnie **„Asystent”**.

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
9. **„Stop teraz”** → aktywne zadania i głos zostają przerwane.

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
