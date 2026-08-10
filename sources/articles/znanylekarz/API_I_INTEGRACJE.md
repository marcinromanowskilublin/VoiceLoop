# ZnanyLekarz (Docplanner) - API i integracje

Data zebrania: 2026-08-10

## Najwazniejsze oficjalne zrodla

- Dokumentacja API: https://integrations.docplanner.com/docs/
- Przewodnik po zasobach API: https://integrations.docplanner.com/guide/api-objects/resources.html
- Artykul help o integracji z MyDr: https://help.docplanner.com/2/doc/maksymalizacja-efektywnosci-dzieki-integracji-znanylekarz-i-mydr
- Artykul help o integracjach zewnetrznych: https://help.docplanner.com/2/doc/integracje-zewnetrzne-jak-korzystac-z-znanylekarz-podczas-integracji-z-zewnetrznym-systemem
- Strona produktowa (rejestracja online): https://pro.znanylekarz.pl/produkty/funkcjonalnosci/rejestracja-online
- Strona produktowa (polaczenie z MyDr): https://pro.znanylekarz.pl/produkty/polacz-systemy-z-mydr-edm

## Co potwierdza dokumentacja API

- API jest REST i dziala pod schematem:
  - `https://www.{domain}/api/v3/integration/{resource}`
- Podstawowe encje:
  - `facility` (placowka), `doctor` (specjalista), `address` (adres lekarza), `service`, `booking`, `calendar`, `slots`.
- Integracja opiera sie na mapowaniu identyfikatorow po obu stronach.
- Dla dostepnosci i rezerwacji kluczowe sa `address_service_id` oraz `slots`.

## Widoczne endpointy (z dokumentacji)

### Facility/Doctor/Address

- `getFacilities`
- `getFacility`
- `getDoctors`
- `getDoctor`
- `getAddresses`
- `getAddress`
- `updateAddress`

### Services i insurance

- `getServices`
- `getAddressServices`
- `addAddressService`
- `getAddressService`
- `updateAddressService`
- `deleteAddressService`
- `getInsuranceProviders`
- `getInsurancePlans`
- `getAddressInsuranceProviders`
- `addAddressInsuranceProvider`
- `updateOrCreateAddressInsuranceProvider`
- `deleteAddressInsuranceProvider`

### Kalendarze i dostepnosc

- `getCalendar`
- `enableCalendar`
- `disableCalendar`
- `getCalendarBreaks`
- `addCalendarBreak`
- `getCalendarBreak`
- `deleteCalendarBreak`
- `getSlots`
- `replaceSlots`
- `bookSlot`
- `deleteSlots`

### Rezerwacje

- `addBooking`
- `getBookings`
- `getBooking`
- `moveBooking`

## Wnioski integracyjne dla VoiceLoop

- ZnanyLekarz moze byc "warstwa marketplace/rezerwacji", a zrodlem grafiku i logiki operacyjnej pozostaje system EDM.
- Dla stabilnosci trzeba utrzymywac mapowanie:
  - `facility_id`, `doctor_id`, `address_id`, `address_service_id`.
- Operacje na terminach najlepiej opierac o `replaceSlots` + kontrola widocznosci uslug.
- Po stronie VoiceLoop warto od razu trzymac slownik:
  - usluga lokalna -> `service_id` (Docplanner)
  - usluga pod adresem -> `address_service_id`.

## Notatki o wlaczeniu integracji (operacyjnie)

- Z helpa wynika, ze klucze API sa aktywowane po stronie Docplanner/ZnanyLekarz i przesylane klientowi.
- Przy aktywnej integracji terminy publikowane na ZnanyLekarz pochodza z systemu zewnetrznego (np. MyDr EDM).
