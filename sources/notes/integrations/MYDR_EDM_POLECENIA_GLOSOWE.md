# MyDr EDM/EDR - bezpieczne polecenia głosowe

Cel: szybkie otwarcie systemu MyDr/EDM i przygotowanie kontekstu pracy bez
automatycznego wystawiania recept, podpisywania certyfikatem ani używania haseł
poza mechanizmami przeglądarki.

## Polecenia główne

```text
otwórz mydr
otwórz edm
otwórz stronę mydr
otwórz stronę edm
przejdź do mydr
uruchom mydr w chrome
```

## Bezpieczny plan wykonania

```text
1. Uruchom Chrome.
2. Otwórz stronę: https://edm.mydr.pl/
3. Jeśli przeglądarka ma zapisane poświadczenia, pozwól użytkownikowi zalogować
   się standardowym mechanizmem przeglądarki.
4. Nie odczytuj haseł i nie zapisuj ich do pamięci długoterminowej.
5. Po wejściu do systemu zatrzymaj się i pozwól użytkownikowi wybrać wizytę lub
   dalsze czynności.
```

## Polecenie do VoiceLoop

```text
Otwórz Chrome i wejdź na stronę MyDr EDM. Nie wystawiaj recepty i nie podpisuj
niczego automatycznie.
```

## Granica bezpieczeństwa

VoiceLoop może:

- otworzyć Chrome,
- wejść na stronę MyDr/EDM,
- pomóc znaleźć właściwe okno,
- przygotować roboczy kontekst,
- przypomnieć użytkownikowi, co trzeba sprawdzić.

VoiceLoop nie powinien automatycznie:

- wystawiać recept,
- podpisywać recept certyfikatem,
- zatwierdzać czynności medyczno-prawnych,
- używać kluczy API do wykonania czynności wymagających decyzji lekarza,
- zapisywać haseł jako trwałej pamięci.

## Komunikat końcowy

```text
Otworzyłem MyDr EDM. Sprawdź dane pacjenta i wykonaj czynności medyczne ręcznie.
```
