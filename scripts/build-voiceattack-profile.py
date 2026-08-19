from __future__ import annotations

import copy
import shutil
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "voiceattack" / "VoiceLoop.vap"
OUTPUT_PATH = ROOT / "voiceattack" / "VoiceLoop-v2.vap"
TEXT_COPY_PATH = ROOT / "voiceattack" / "VoiceLoop-v2.vap.txt"
PROFILE_ID = uuid.UUID("40125479-3df4-4d50-a0cb-1f9450b8fc6b")
ID_NAMESPACE = uuid.UUID("bb9df565-c7dd-4ae4-a8ed-aa20aa14fa79")
VOICE_RECOGNITION_CONFIDENCE = 65


@dataclass(frozen=True)
class VoiceCommand:
    key: str
    phrases: str
    script: str
    description: str


def phrase_pack(*groups: str) -> str:
    phrases: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_phrase in group.split(";"):
            phrase = " ".join(raw_phrase.strip().split())
            normalized = phrase.casefold()
            if phrase and normalized not in seen:
                seen.add(normalized)
                phrases.append(phrase)
    return ";".join(phrases)


COMMANDS = (
    VoiceCommand(
        "assistant",
        phrase_pack(
            "asystent;asystencie;hej asystent;hej asystencie;halo asystent",
            "halo asystencie;voice loop;voiceloop;słuchaj asystencie",
            "sluchaj asystencie;posłuchaj asystencie;posluchaj asystencie",
            "mam polecenie;mam pytanie;chcę o coś zapytać;chce o cos zapytac",
            "uruchom asystenta;włącz asystenta;wlacz asystenta;asys",
        ),
        "assistant.vbs",
        "Jednorazowy nasłuch Deepgram i przekazanie wypowiedzi do asystenta.",
    ),
    VoiceCommand(
        "note",
        phrase_pack(
            "zapisz notatkę;zapisz notatke;zapisz mi notatkę;zapisz mi notatke",
            "nowa notatka;utwórz notatkę;utworz notatke;dodaj notatkę",
            "dodaj notatke;zrób notatkę;zrob notatke;stwórz notatkę",
            "stworz notatke;zanotuj to;zanotuj;dopisz notatkę;dopisz notatke",
            "zapisz to jako notatkę;zapisz to jako notatke;notatka;notka",
        ),
        "note.vbs",
        "Pyta o treść i zapisuje notatkę przez kontrolowaną akcję.",
    ),
    VoiceCommand(
        "remember",
        phrase_pack(
            "zapamiętaj;zapamietaj;zapamiętaj to;zapamietaj to",
            "zapamiętaj to proszę;zapamietaj to prosze;pamiętaj to;pamietaj to",
            "zapisz to w pamięci;zapisz to w pamieci;dodaj to do pamięci",
            "dodaj to do pamieci;zachowaj to w pamięci;zachowaj to w pamieci",
            "nie zapomnij tego;chcę żebyś to pamiętał;chce zebys to pamietal",
            "pamiętaj;pamietaj",
        ),
        "remember.vbs",
        "Pyta o fakt, a zapis wykonuje dopiero po potwierdzeniu.",
    ),
    VoiceCommand(
        "recent-activity",
        phrase_pack(
            "co robiłem ostatnio;co robilem ostatnio;co robiłam ostatnio",
            "co robilam ostatnio;co ostatnio robiłem;co ostatnio robilem",
            "ostatnia aktywność;ostatnia aktywnosc;moja ostatnia aktywność",
            "moja ostatnia aktywnosc;co się działo na ekranie;co sie dzialo na ekranie",
            "pokaż historię ekranu;pokaz historie ekranu;pokaż historię aktywności",
            "pokaz historie aktywnosci;podsumuj ostatnią aktywność",
            "podsumuj ostatnia aktywnosc;podsumuj co robiłem;podsumuj co robilem",
            "co było na ekranie;co bylo na ekranie;aktywność;aktywnosc;historia",
        ),
        "recent-activity.vbs",
        "Czyta lokalne podsumowanie ostatniej aktywności Screenpipe.",
    ),
    VoiceCommand(
        "active-window",
        phrase_pack(
            "opisz aktywne okno;opisz bieżące okno;opisz biezace okno",
            "co mam otwarte;co jest teraz otwarte;jakie okno jest aktywne",
            "które okno jest aktywne;ktore okno jest aktywne",
            "na jakim oknie jestem;w jakim oknie jestem;gdzie teraz jestem",
            "podaj aktywne okno;sprawdź aktywne okno;sprawdz aktywne okno",
            "jaki program jest aktywny;co jest na wierzchu;aktywne okno",
        ),
        "active-window.vbs",
        "Czyta tytuł aktywnego okna i nazwę programu.",
    ),
    VoiceCommand(
        "check-text-target",
        phrase_pack(
            "gdzie teraz piszę;gdzie teraz pisze;sprawdź pole tekstowe",
            "sprawdz pole tekstowe;sprawdź gdzie piszę;sprawdz gdzie pisze",
            "gdzie trafi tekst;gdzie wpisze się tekst;gdzie wpisze sie tekst",
            "do jakiego pola trafi tekst;czy to dobre pole;czy mogę tu pisać",
            "czy moge tu pisac;czy to pasek adresu;sprawdź kursor tekstowy",
            "sprawdz kursor tekstowy;sprawdź miejsce wpisywania",
            "sprawdz miejsce wpisywania;pole tekstowe",
        ),
        "check-text-target.vbs",
        "Sprawdza, czy wpisywanie trafi do właściwego pola tekstowego.",
    ),
    VoiceCommand(
        "minimize-window",
        phrase_pack(
            "zminimalizuj okno;minimalizuj okno;zminimalizuj aktywne okno",
            "minimalizuj aktywne okno;schowaj okno;schowaj aktywne okno",
            "ukryj okno;ukryj aktywne okno;zwiń okno;zwin okno",
            "zwiń aktywne okno;zwin aktywne okno;opuść to okno",
            "opusc to okno;minimalizuj to;schowaj to okno;zwiń;zwin",
        ),
        "minimize-window.vbs",
        "Minimalizuje aktualnie aktywne okno.",
    ),
    VoiceCommand(
        "minimize-window-under-cursor",
        phrase_pack(
            "zminimalizuj okno pod kursorem;minimalizuj okno pod kursorem",
            "zminimalizuj aplikację pod kursorem;zminimalizuj aplikacje pod kursorem",
            "minimalizuj aplikację pod kursorem;minimalizuj aplikacje pod kursorem",
            "schowaj okno pod kursorem;ukryj okno pod kursorem",
            "zwiń okno pod kursorem;zwin okno pod kursorem",
            "zminimalizuj okno pod myszką;zminimalizuj okno pod myszka",
        ),
        "minimize-window-under-cursor.vbs",
        "Minimalizuje okno aplikacji wskazywane kursorem.",
    ),
    VoiceCommand(
        "minimize-all",
        phrase_pack(
            "zminimalizuj wszystkie;zminimalizuj wszystkie okna",
            "minimalizuj wszystkie okna;minimalizuj wszystko",
            "schowaj wszystkie okna;ukryj wszystkie okna",
            "zwiń wszystkie okna;zwin wszystkie okna;zwiń wszystko;zwin wszystko",
            "pokaż pulpit;pokaz pulpit;przejdź do pulpitu;przejdz do pulpitu",
            "odsłoń pulpit;odslon pulpit;wyczyść ekran;wyczysc ekran;pulpit",
        ),
        "minimize-all.vbs",
        "Minimalizuje wszystkie okna i pokazuje pulpit.",
    ),
    VoiceCommand(
        "close-window-under-cursor",
        phrase_pack(
            "zamknij okno pod kursorem;zamknij wskazane okno",
            "zamknij okno które wskazuję;zamknij okno ktore wskazuje",
            "zamknij okno pod myszką;zamknij okno pod myszka",
            "zamknij aplikację pod kursorem;zamknij aplikacje pod kursorem",
            "wyłącz aplikację pod kursorem;wylacz aplikacje pod kursorem",
            "zamknij aplikację pod myszką;zamknij aplikacje pod myszka",
            "wyłącz aplikację pod myszką;wylacz aplikacje pod myszka",
            "zamknij wskazaną aplikację;zamknij wskazana aplikacje",
            "wyłącz wskazaną aplikację;wylacz wskazana aplikacje",
            "zamknij program pod kursorem;wyłącz program pod kursorem",
            "wylacz program pod kursorem;zamknij wskazany program",
            "zamknij program który wskazuję;zamknij program ktory wskazuje",
            "zamknij to okno;wyłącz tę aplikację;wylacz te aplikacje",
            "zamknij tę aplikację;zamknij te aplikacje",
        ),
        "close-window-under-cursor.vbs",
        "Prosi wskazane okno o zamknięcie po potwierdzeniu głosowym.",
    ),
    VoiceCommand(
        "copy-selected-text",
        phrase_pack(
            "kopiuj zaznaczony tekst;skopiuj zaznaczony tekst",
            "kopiuj zaznaczony fragment;skopiuj zaznaczony fragment",
            "kopiuj to co zaznaczyłem;kopiuj to co zaznaczylem",
            "skopiuj to co zaznaczyłem;skopiuj to co zaznaczylem",
            "wrzuć zaznaczenie do schowka;wrzuc zaznaczenie do schowka",
            "przenieś zaznaczenie do schowka;przenies zaznaczenie do schowka",
            "przepisz zaznaczenie do schowka;zaznaczony tekst do schowka",
            "zaznaczenie do schowka;kopiuj zaznaczenie;skopiuj zaznaczenie",
        ),
        "copy-selected-text.vbs",
        "Kopiuje bieżący zaznaczony tekst do schowka.",
    ),
    VoiceCommand(
        "copy-text-under-cursor",
        phrase_pack(
            "kopiuj tekst pod kursorem;skopiuj tekst pod kursorem",
            "kopiuj tekst pod myszką;kopiuj tekst pod myszka",
            "skopiuj tekst pod myszką;skopiuj tekst pod myszka",
            "kopiuj spod kursora;skopiuj spod kursora",
            "kopiuj pod kursorem;skopiuj pod kursorem",
            "kopiuj to pod kursorem;skopiuj to pod kursorem",
            "kopiuj element pod kursorem;skopiuj element pod kursorem",
            "kopiuj wskazany tekst;skopiuj wskazany tekst",
            "kopiuj tekst który wskazuję;kopiuj tekst ktory wskazuje",
            "skopiuj tekst który wskazuję;skopiuj tekst ktory wskazuje",
            "kopiuj to co wskazuje myszka;skopiuj to co wskazuje myszka",
            "tekst pod kursorem do schowka;tekst pod myszką do schowka",
            "tekst pod myszka do schowka;wskazany tekst do schowka",
        ),
        "copy-text-under-cursor.vbs",
        "Kopiuje dostępny tekst elementu wskazywanego kursorem.",
    ),
    VoiceCommand(
        "copy-email-under-cursor",
        phrase_pack(
            "kopiuj email pod kursorem;skopiuj email pod kursorem",
            "kopiuj e-mail pod kursorem;skopiuj e-mail pod kursorem",
            "kopiuj mail pod kursorem;skopiuj mail pod kursorem",
            "kopiuj adres email pod kursorem;skopiuj adres email pod kursorem",
            "kopiuj adres e-mail pod kursorem;skopiuj adres e-mail pod kursorem",
            "kopiuj adres mailowy pod kursorem;skopiuj adres mailowy pod kursorem",
            "kopiuj maila pod kursorem;skopiuj maila pod kursorem",
            "kopiuj email pod myszką;kopiuj email pod myszka",
            "skopiuj email pod myszką;skopiuj email pod myszka",
            "kopiuj mail pod myszką;kopiuj mail pod myszka",
            "skopiuj mail pod myszką;skopiuj mail pod myszka",
            "skopiuj mail który wskazuję;skopiuj mail ktory wskazuje",
            "skopiuj email który wskazuję;skopiuj email ktory wskazuje",
            "skopiuj mi ten mail gdzie mam kursor do schowka",
            "skopiuj ten email gdzie jest kursor;adres email do schowka",
            "adres e-mail do schowka;mail do schowka;email do schowka",
            "kopiuj email;skopiuj email;kopiuj mail;skopiuj mail",
        ),
        "copy-email-under-cursor.vbs",
        "Kopiuje pojedynczy adres e-mail wykryty pod kursorem.",
    ),
    VoiceCommand(
        "copy-number-under-cursor",
        phrase_pack(
            "kopiuj numer pod kursorem;skopiuj numer pod kursorem",
            "kopiuj liczbę pod kursorem;kopiuj liczbe pod kursorem",
            "skopiuj liczbę pod kursorem;skopiuj liczbe pod kursorem",
            "kopiuj telefon pod kursorem;skopiuj telefon pod kursorem",
            "kopiuj numer telefonu pod kursorem;skopiuj numer telefonu pod kursorem",
            "kopiuj numer pod myszką;kopiuj numer pod myszka",
            "skopiuj numer pod myszką;skopiuj numer pod myszka",
            "kopiuj liczbę którą wskazuję;kopiuj liczbe ktora wskazuje",
            "skopiuj numer który wskazuję;skopiuj numer ktory wskazuje",
            "numer pod kursorem do schowka;telefon pod kursorem do schowka",
            "numer do schowka;kopiuj numer;skopiuj numer",
        ),
        "copy-number-under-cursor.vbs",
        "Kopiuje numer wykryty pod kursorem.",
    ),
    VoiceCommand(
        "copy-sentence-under-cursor",
        phrase_pack(
            "kopiuj całe zdanie pod kursorem;kopiuj cale zdanie pod kursorem",
            "skopiuj całe zdanie pod kursorem;skopiuj cale zdanie pod kursorem",
            "kopiuj zdanie pod kursorem;skopiuj zdanie pod kursorem",
            "kopiuj całe zdanie pod myszką;kopiuj cale zdanie pod myszka",
            "skopiuj zdanie pod myszką;skopiuj zdanie pod myszka",
            "kopiuj zdanie które wskazuję;kopiuj zdanie ktore wskazuje",
            "zdanie pod kursorem do schowka",
            "całe zdanie do schowka;cale zdanie do schowka",
            "kopiuj zdanie;skopiuj zdanie",
        ),
        "copy-sentence-under-cursor.vbs",
        "Kopiuje całe zdanie wykryte pod kursorem.",
    ),
    VoiceCommand(
        "select-sentence-under-cursor",
        phrase_pack(
            "zaznacz zdanie;zaznacz całe zdanie;zaznacz cale zdanie",
            "zaznacz zdanie pod kursorem;zaznacz całe zdanie pod kursorem",
            "zaznacz cale zdanie pod kursorem",
            "zaznacz zdanie pod myszką;zaznacz zdanie pod myszka",
            "zaznacz całe zdanie pod myszką;zaznacz cale zdanie pod myszka",
            "zaznacz wskazane zdanie;zaznacz to zdanie",
            "zaznacz zdanie które wskazuję;zaznacz zdanie ktore wskazuje",
            "wybierz zdanie pod kursorem;wybierz całe zdanie",
            "wybierz cale zdanie;wybierz wskazane zdanie",
            "podświetl zdanie pod kursorem;podswietl zdanie pod kursorem",
            "podświetl całe zdanie;podswietl cale zdanie",
        ),
        "select-sentence-under-cursor.vbs",
        "Zaznacza zdanie pod kursorem przez Windows UI Automation.",
    ),
    VoiceCommand(
        "select-paragraph-under-cursor",
        phrase_pack(
            "zaznacz akapit;zaznacz cały akapit;zaznacz caly akapit",
            "zaznacz akapit pod kursorem;zaznacz cały akapit pod kursorem",
            "zaznacz caly akapit pod kursorem",
            "zaznacz akapit pod myszką;zaznacz akapit pod myszka",
            "zaznacz cały akapit pod myszką;zaznacz caly akapit pod myszka",
            "zaznacz wskazany akapit;zaznacz ten akapit",
            "zaznacz akapit który wskazuję;zaznacz akapit ktory wskazuje",
            "wybierz akapit pod kursorem;wybierz cały akapit",
            "wybierz caly akapit;wybierz wskazany akapit",
            "podświetl akapit pod kursorem;podswietl akapit pod kursorem",
            "podświetl cały akapit;podswietl caly akapit",
            "zaznacz paragraf;zaznacz paragraf pod kursorem",
        ),
        "select-paragraph-under-cursor.vbs",
        "Zaznacza akapit pod kursorem przez Windows UI Automation.",
    ),
    VoiceCommand(
        "listen-start",
        phrase_pack(
            "włącz nasłuch;wlacz nasluch;uruchom nasłuch;uruchom nasluch",
            "zacznij nasłuch;zacznij nasluch;start nasłuchu;start nasluchu",
            "słuchaj ciągle;sluchaj ciagle;słuchaj cały czas;sluchaj caly czas",
            "włącz mikrofon asystenta;wlacz mikrofon asystenta",
            "włącz tryb rozmowy;wlacz tryb rozmowy;włącz rozmowę;wlacz rozmowe",
            "nasłuch włączony;nasluch wlaczony;nasłuch on",
        ),
        "listen-start.vbs",
        "Włącza ciągły nasłuch Deepgram.",
    ),
    VoiceCommand(
        "listen-stop",
        phrase_pack(
            "wyłącz nasłuch;wylacz nasluch;zatrzymaj nasłuch;zatrzymaj nasluch",
            "przerwij nasłuch;przerwij nasluch;stop nasłuchu;stop nasluchu",
            "przestań słuchać;przestan sluchac;wyłącz mikrofon asystenta",
            "wylacz mikrofon asystenta;wyłącz tryb rozmowy;wylacz tryb rozmowy",
            "wyłącz rozmowę;wylacz rozmowe;nasłuch wyłączony",
            "nasluch wylaczony;nasłuch off;nie słuchaj;nie sluchaj",
        ),
        "listen-stop.vbs",
        "Wyłącza Deepgram bez anulowania wykonywanych akcji.",
    ),
    VoiceCommand(
        "status",
        phrase_pack(
            "status voice loop;status voiceloop;status asystenta",
            "czy działasz;czy dzialasz;czy voice loop działa;czy voice loop dziala",
            "jaki jest status;jaki masz status;podaj status systemu",
            "podaj status asystenta;sprawdź status;sprawdz status",
            "sprawdź czy wszystko działa;sprawdz czy wszystko dziala",
            "czy system jest gotowy;czy asystent jest gotowy",
            "jak stoisz;stan systemu;stan asystenta;status",
        ),
        "status.vbs",
        "Czyta krótki stan rdzenia, modelu i nasłuchu.",
    ),
    VoiceCommand(
        "confirm",
        phrase_pack(
            "potwierdź;potwierdz;tak potwierdzam;tak potwierdź",
            "tak potwierdz;zatwierdź;zatwierdz;zatwierdź zadanie",
            "zatwierdz zadanie;potwierdź wykonanie;potwierdz wykonanie",
            "wykonaj to;możesz wykonać;mozesz wykonac;zgadzam się",
            "zgadzam sie;masz zgodę;masz zgode;tak zrób to;tak zrob to",
            "kontynuuj wykonanie;uruchom zadanie",
        ),
        "confirm.vbs",
        "Potwierdza najnowsze polecenie oczekujące na zgodę.",
    ),
    VoiceCommand(
        "cancel",
        phrase_pack(
            "anuluj zadanie;anuluj polecenie;anuluj wykonanie",
            "nie rób tego;nie rob tego;jednak nie;wycofaj polecenie",
            "wycofaj zadanie;odrzuć polecenie;odrzuc polecenie",
            "odrzuć zadanie;odrzuc zadanie;przerwij zadanie",
            "nie wykonuj tego;nie wykonuj zadania;cofnij polecenie",
            "rezygnuję z zadania;rezygnuje z zadania;anuluj",
        ),
        "cancel.vbs",
        "Anuluje najnowsze polecenie oczekujące na zgodę.",
    ),
    VoiceCommand(
        "voice-test",
        phrase_pack(
            "test pętli;test petli;test głosu;test glosu;voice test",
            "test voice loop;test voiceloop;sprawdź głos;sprawdz glos",
            "sprawdź pętlę;sprawdz petle;przetestuj asystenta",
            "test asystenta;czy mnie słyszysz;czy mnie slyszysz;test",
        ),
        "voice-test.vbs",
        "Sprawdza pełną lokalną pętlę VoiceLoop.",
    ),
    VoiceCommand(
        "open-calendar",
        phrase_pack(
            "otwórz kalendarz;otworz kalendarz;open calendar",
            "uruchom kalendarz;włącz kalendarz;wlacz kalendarz",
            "pokaż kalendarz;pokaz kalendarz;przejdź do kalendarza",
            "przejdz do kalendarza;chcę zobaczyć kalendarz",
            "chce zobaczyc kalendarz;wyświetl kalendarz;wyswietl kalendarz",
            "odpal kalendarz;otwórz mój kalendarz;otworz moj kalendarz;kalendarz",
        ),
        "open-calendar.vbs",
        "Otwiera kalendarz przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "open-browser",
        phrase_pack(
            "otwórz przeglądarkę;otworz przegladarke;open browser",
            "uruchom przeglądarkę;uruchom przegladarke",
            "włącz przeglądarkę;wlacz przegladarke",
            "pokaż przeglądarkę;pokaz przegladarke",
            "przejdź do przeglądarki;przejdz do przegladarki",
            "otwórz internet;otworz internet;odpal przeglądarkę",
            "odpal przegladarke;nowe okno przeglądarki",
            "nowe okno przegladarki;przeglądarka;przegladarka",
        ),
        "open-browser.vbs",
        "Otwiera przeglądarkę przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "search-web",
        phrase_pack(
            "wyszukaj w internecie;wyszukaj w necie;poszukaj w internecie",
            "poszukaj w necie;sprawdź w internecie;sprawdz w internecie",
            "sprawdź w necie;sprawdz w necie;szukaj online",
            "wyszukaj online;zrób wyszukiwanie;zrob wyszukiwanie",
            "przeszukaj internet;przeszukaj sieć;przeszukaj siec",
            "znajdź w internecie;znajdz w internecie",
            "sprawdź to online;sprawdz to online;wyszukiwanie internetowe",
        ),
        "search-web.vbs",
        "Uruchamia szybkie wyszukiwanie informacji w internecie.",
    ),
    VoiceCommand(
        "open-chat",
        phrase_pack(
            "otwórz czat;otworz czat;open chat;uruchom czat",
            "włącz czat;wlacz czat;pokaż czat;pokaz czat",
            "przejdź do czatu;przejdz do czatu;nowy czat;czat",
        ),
        "open-chat.vbs",
        "Otwiera czat przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "open-gpt-chat",
        phrase_pack(
            "otwórz gpt;otworz gpt;open gpt;uruchom gpt",
            "włącz gpt;wlacz gpt;pokaż gpt;pokaz gpt",
            "otwórz chat gpt;otworz chat gpt;otwórz chatgpt",
            "otworz chatgpt;uruchom chat gpt;uruchom chatgpt",
            "nowy chat gpt;nowy chatgpt;nowy czat gpt",
            "przejdź do chat gpt;przejdz do chat gpt;chatgpt",
        ),
        "open-gpt-chat.vbs",
        "Otwiera osobno czat GPT.",
    ),
    VoiceCommand(
        "open-gemini-chat",
        phrase_pack(
            "otwórz gemini;otworz gemini;open gemini;uruchom gemini",
            "włącz gemini;wlacz gemini;pokaż gemini;pokaz gemini",
            "otwórz czat gemini;otworz czat gemini;czat gemini",
            "nowy czat gemini;nowy chat gemini;przejdź do gemini",
            "przejdz do gemini;uruchom czat gemini;gemini chat;gemini",
        ),
        "open-gemini-chat.vbs",
        "Otwiera osobno czat Gemini.",
    ),
    VoiceCommand(
        "remember-last-source",
        phrase_pack(
            "zapamiętaj ostatnie źródło;zapamietaj ostatnie zrodlo",
            "zapisz ostatnie źródło;zapisz ostatnie zrodlo",
            "zachowaj ostatnie źródło;zachowaj ostatnie zrodlo",
            "zapamiętaj ostatni link;zapamietaj ostatni link",
            "zapisz ostatni link;zachowaj ostatni link",
            "zapamiętaj wynik wyszukiwania;zapamietaj wynik wyszukiwania",
            "zapisz wynik wyszukiwania;zachowaj wynik wyszukiwania",
            "zapamiętaj ostatni wynik;zapamietaj ostatni wynik",
            "zapisz ostatni wynik;zachowaj ostatni wynik",
            "dodaj źródło do pamięci;dodaj zrodlo do pamieci",
            "zapamiętaj to źródło;zapamietaj to zrodlo",
            "zapisz to źródło;zapisz to zrodlo",
        ),
        "remember-last-source.vbs",
        "Zapisuje ostatnie źródło internetowe po potwierdzeniu.",
    ),
    VoiceCommand(
        "capabilities",
        phrase_pack(
            "co potrafisz;co umiesz;jakie masz komendy;jakie znasz komendy",
            "jakie masz akcje;pokaż możliwości;pokaz mozliwosci",
            "wymień możliwości;wymien mozliwosci;pomoc głosowa;pomoc glosowa",
        ),
        "capabilities.vbs",
        "Czyta możliwości VoiceAttack i dodatkowe własne akcje VoiceLoop.",
    ),
    VoiceCommand(
        "rename-under-cursor",
        phrase_pack(
            "zmień nazwę;zmien nazwe;zmień nazwę pod kursorem;zmien nazwe pod kursorem",
            "przemianuj;przemianuj pod kursorem;zmień nazwę ikony;zmien nazwe ikony",
            "rename under cursor;zmień nazwę pliku;zmien nazwe pliku",
        ),
        "rename-under-cursor.vbs",
        "Zmienia nazwę elementu pod kursorem po potwierdzeniu (Deepgram/VA → VoiceLoop).",
    ),
    VoiceCommand(
        "stop",
        phrase_pack(
            "stop teraz;zatrzymaj wszystko;przerwij wszystko",
            "natychmiastowy stop;natychmiast zatrzymaj;awaryjnie stop",
            "awaryjne zatrzymanie;panic stop;panic button;abort",
            "anuluj wszystko;wyłącz wszystko;wylacz wszystko",
            "zatrzymaj voice loop;zatrzymaj voiceloop",
            "przerwij voice loop;przerwij voiceloop;stop wszystko;stop",
        ),
        "stop.vbs",
        "Natychmiast zatrzymuje nasłuch, kolejkę, TTS i bieżącą akcję.",
    ),
)


def set_text(element: ET.Element, name: str, value: str) -> None:
    target = element.find(name)
    if target is None:
        raise RuntimeError(f"Szablon VoiceAttack nie zawiera elementu {name}.")
    target.text = value


def validate_commands() -> int:
    phrase_owners: dict[str, str] = {}
    phrase_count = 0
    for definition in COMMANDS:
        script_path = ROOT / "scripts" / "va" / definition.script
        if not script_path.is_file():
            raise RuntimeError(f"Brak skryptu komendy {definition.key}: {script_path}")
        for phrase in definition.phrases.split(";"):
            normalized = " ".join(phrase.casefold().split())
            previous_owner = phrase_owners.get(normalized)
            if previous_owner is not None:
                raise RuntimeError(
                    f"Fraza '{phrase}' jest wspólna dla {previous_owner} i {definition.key}."
                )
            phrase_owners[normalized] = definition.key
            phrase_count += 1
    return phrase_count


def build_profile() -> None:
    phrase_count = validate_commands()
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    ET.register_namespace("xsd", "http://www.w3.org/2001/XMLSchema")
    tree = ET.parse(TEMPLATE_PATH)
    root = tree.getroot()
    commands_node = root.find("Commands")
    if commands_node is None or not list(commands_node):
        raise RuntimeError("Szablon VoiceAttack nie zawiera przykładowej komendy.")

    template_command = copy.deepcopy(next(iter(commands_node)))
    commands_node.clear()
    set_text(root, "Id", str(PROFILE_ID))
    set_text(root, "Name", "VoiceLoop v2 PRO")

    last_command_id = ""
    for definition in COMMANDS:
        command = copy.deepcopy(template_command)
        command_id = str(uuid.uuid5(ID_NAMESPACE, f"command:{definition.key}"))
        action_id = str(uuid.uuid5(ID_NAMESPACE, f"action:{definition.key}"))
        last_command_id = command_id

        set_text(command, "Id", command_id)
        set_text(command, "CommandString", definition.phrases)
        set_text(command, "Description", f"VoiceLoop v2 PRO: {definition.description}")
        set_text(command, "Category", "VoiceLoop v2 PRO")
        set_text(command, "Async", "true")
        set_text(command, "lastEditedAction", action_id)
        set_text(command, "ExecFromWildcard", "false")
        set_text(command, "UseConfidence", "true")
        set_text(command, "minimumConfidenceLevel", str(VOICE_RECOGNITION_CONFIDENCE))

        action = command.find("./ActionSequence/CommandAction")
        if action is None:
            raise RuntimeError("Szablon VoiceAttack nie zawiera akcji uruchomienia.")
        set_text(action, "Id", action_id)
        set_text(action, "ActionType", "Launch")
        set_text(
            action,
            "Context",
            str(ROOT / "scripts" / "va" / definition.script),
        )
        commands_node.append(command)

    set_text(root, "LastEditedCommand", last_command_id)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_PATH, encoding="utf-8", xml_declaration=True)
    shutil.copyfile(OUTPUT_PATH, TEXT_COPY_PATH)
    print(
        f"Zbudowano {OUTPUT_PATH} "
        f"({len(COMMANDS)} komend, {phrase_count} wariantów fraz)."
    )


if __name__ == "__main__":
    build_profile()
