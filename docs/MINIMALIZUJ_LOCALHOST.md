# VoiceLoop Minimalizuj — localhost studio

## Uruchomienie teraz

Otwórz PowerShell w katalogu repozytorium (Windows, Python 3.11):

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-minimalizuj.ps1

Panel: http://127.0.0.1:8792/minimalizuj

Health: http://127.0.0.1:8792/api/health

## Autostart po zalogowaniu

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-minimalizuj-autostart.ps1

Powstanie zadanie użytkownika "VoiceLoop Minimalizuj" (bez uprawnień SYSTEM).
Watchdog wznawia proces po jego awarii z pięciosekundową przerwą.
Nie działa przed logowaniem. Aby wyłączyć autostart:

    Unregister-ScheduledTask -TaskName "VoiceLoop Minimalizuj" -Confirm:$false

## Jak kalibrować

1. Spacja: nagrywaj. Spacja: zatrzymaj i zapisz lokalnie.
2. Uzupełnij frazę rozpoznaną przez STT albo wpisz ją ręcznie.
3. Popraw frazę i porównaj różnicę. Wybierz opcjonalnie zatwierdzenie aliasu.
4. Próbka zapisuje się w data/calibration-phrases/minimalizuj.sqlite3;
   audio obok, w tym samym katalogu.
5. Zatwierdzone aliasy obowiązują WYŁĄCZNIE w kalibracyjnym studio.
   Nie są automatycznie publikowane w produkcyjnym routerze ani wykonywane.
6. Osobny przycisk Wykonaj minimalizację wymaga potwierdzenia w przeglądarce.
   Serwer 8792 przekazuje wyłącznie zarejestrowaną akcję do rdzenia 8765,
   z lokalnym tokenem sesji, i pokazuje późniejszy status executora.
   Nie ma uniwersalnego endpointu wykonywania arbitralnych poleceń.
   Ta gałąź dodaje do routera jednoznaczny alias "minimalizuj".
   Kliknięcie bez działającego rdzenia 8765 zwróci czytelny błąd.

STT przeglądarkowe jest OPCJONALNE i może przesyłać dźwięk do zewnętrznego dostawcy;
nie używaj go do prywatnych rozmów bez odpowiedniej podstawy i zgody.
Nie ma gwarancji działania ani jakości STT bez dostępnego dostawcy.
Samo nagrywanie przeglądarkowe działa lokalnie.

## Trwałość

- Trwały katalog SQLite nie znajduje się w Git; zachowuje dane po restarcie.
- Zadanie działa w sesji aktualnie zalogowanego użytkownika.
- Nawet działający watchdog nie gwarantuje, że mikrofon jest dostępny.
- W razie problemu sprawdź logs/minimalizuj.stderr.log oraz /api/health.
- Port 8792 może obsługiwać tylko jeden serwer naraz. Nie uruchamiaj równolegle
  starej instancji kalibratora.
- Serwer dopuszcza wyłącznie localhost w nagłówku Host i Origin.
- Zapis aliasu nie trenuje modelu STT; to korekta warstwy interpretacji.
