# Mapa integracji: ZnanyLekarz <-> mydr EDM

Data: 2026-08-10

## Cel

Zebranie sensownych materialow pod wdrozenie integracji VoiceLoop z:

- ZnanyLekarz (Docplanner)
- mydr EDM

## Gdzie zapisane materialy

- ZnanyLekarz:
  - `sources/articles/znanylekarz/API_I_INTEGRACJE.md`
- mydr EDM:
  - `sources/articles/mydr_edm/API_I_INTEGRACJE.md`

## Najwazniejsze obserwacje

- ZnanyLekarz ma publicznie dostepna, konkretna dokumentacje integracyjna.
- mydr EDM: oficjalne API docs byly chwilowo niedostepne (awaria strony) podczas zbierania.
- Da sie juz ulozyc robocza mape endpointow i przeplywu auth, ale przed wdrozeniem produkcyjnym potrzebna jest ponowna walidacja oficjalnych docs mydr.

## Proponowany porzadek wdrozenia (bez kodu, plan roboczy)

1. Potwierdzic dzialanie docs mydr po stronie producenta.
2. Uzgodnic scope MVP:
   - pobieranie wolnych terminow,
   - tworzenie wizyt,
   - synchronizacja statusow.
3. Zrobic mapowanie ID:
   - lekarz, adres, usluga, slot po obu stronach.
4. Dodac monitor integralnosci:
   - rozjazdy slotow,
   - konflikt statusow wizyt,
   - niedostepnosc API.

## Kontakty i onboarding (z materialow)

- ZnanyLekarz:
  - wsparcie biznesowe/techniczne przez centrum pomocy i dedykowane adresy (np. `cc@znanylekarz.pl`, `placowki@znanylekarz.pl` zaleznie od typu konta).
- mydr:
  - `kontakt@mydr.pl`
  - `https://pro.mydr.pl/kontakt`

## Uwaga o jakosci danych

Nie kopiowalem przypadkowych blogow ani agregatorow. Material oparty na:

- oficjalnych docs i helpach Docplanner/ZnanyLekarz,
- oficjalnej stronie API mydr,
- snapshotach endpointow tylko jako wsparcie tymczasowe przy awarii docs.
