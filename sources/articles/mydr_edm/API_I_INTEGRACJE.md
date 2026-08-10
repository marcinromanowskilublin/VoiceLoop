# mydr EDM - API i integracje

Data zebrania: 2026-08-10

## Najwazniejsze zrodla

- Strona API mydr: https://api.edm.mydr.pl/
- API docs (aktualnie niedostepne): https://api.edm.mydr.pl/api-docs/
- OAuth flow (fake client): https://api.edm.mydr.pl/api-docs/fake-oauth-client/exchange/
- Snapshot endpointow (zrodlo pomocnicze): https://context7.com/websites/api_edm_mydr_pl_api-docs/llms.txt
- Strona kontaktu komercyjnego: https://pro.mydr.pl/kontakt
- Integracja ZnanyLekarz + mydr: https://pro.znanylekarz.pl/produkty/polacz-systemy-z-mydr-edm

## Status dostepnosci API (w momencie zbierania)

- `api.edm.mydr.pl` zwraca komunikat o utrudnieniach logowania (awaria po stronie uslugi).
- `api.edm.mydr.pl/api-docs/` takze zwraca komunikat awaryjny.
- To oznacza, ze czesc techniczna ponizej opiera sie na:
  - oficjalnej stronie API mydr (opis funkcji),
  - materialach pomocniczych i snapshotach endpointow.

## Co potwierdza oficjalna strona API mydr

- Integracja sluzy do:
  - tworzenia wizyt,
  - tworzenia pacjentow,
  - sprawdzania wolnych terminow,
  - e-recept/e-skierowan,
  - weryfikacji ubezpieczenia.
- Dostep do API:
  - w panelu placowki: `Ustawienia placowki -> Abonament -> Uslugi dodatkowe -> Dostep do API`.
- Kontakt:
  - `kontakt@mydr.pl`
  - formularz: `https://pro.mydr.pl/kontakt`

## Endpointy/zasoby wykryte w dostepnych materialach

Uwaga: lista robocza do weryfikacji po powrocie oficjalnego API docs.

- `GET /secure/ext_api/patients/{patient_pk}/visits/`
- `GET /secure/ext_api/doctors/`
- `GET /secure/ext_api/visits/free_slots/`
- `GET /secure/ext_api/visits/free_slots_v2/`
- `POST /secure/ext_api/prescriptions/new_draft_straight/`
- `POST /secure/ext_api/prescriptions/`
- `GET /secure/ext_api/medicines/search/`
- `POST /secure/ext_api/nurses/`
- `GET /secure/ext_api/visits/{visit_pk}/services/`
- `GET /secure/ext_api/visits/{visit_pk}/diagnoses/`

## OAuth i tokeny (na podstawie materialu fake oauth)

- Przeplyw: authorization code -> access token.
- Token response zawiera:
  - `access_token`
  - `refresh_token`
  - `expires_in`
- W materiale testowym opisano tez odswiezanie tokena (`refresh_token`).
- Authorization code ma krotki czas waznosci (opisowo ok. 10 minut), a access token ma TTL rzedu godzin (opisowo ok. 10h).

## Wnioski integracyjne dla VoiceLoop

- Integracja z mydr powinna miec osobna warstwe:
  - autoryzacja OAuth,
  - adapter endpointow wizyt/pacjentow,
  - mapper uslug i lekarzy.
- Poniewaz oficjalne docs byly niedostepne podczas audytu:
  - przed implementacja produkcyjna trzeba wykonac ponowny crawl docs i potwierdzic endpointy/schematy.
