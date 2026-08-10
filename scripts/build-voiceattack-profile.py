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


@dataclass(frozen=True)
class VoiceCommand:
    key: str
    phrases: str
    script: str
    description: str


COMMANDS = (
    VoiceCommand(
        "assistant",
        "asystent;hej asystent;voice loop;sluchaj asystencie;mam polecenie;posluchaj;asys",
        "assistant.vbs",
        "Jednorazowy nasłuch Deepgram i przekazanie wypowiedzi do asystenta.",
    ),
    VoiceCommand(
        "note",
        "zapisz notatkę;nowa notatka;utworz notatke;dodaj notatke;zanotuj to;notatka;notka",
        "note.vbs",
        "Pyta o treść i zapisuje notatkę przez kontrolowaną akcję.",
    ),
    VoiceCommand(
        "remember",
        "zapamiętaj;zapamiętaj to;zapamietaj to prosze;pamietaj to;zapisz to w pamieci;pamietaj",
        "remember.vbs",
        "Pyta o fakt, a zapis wykonuje dopiero po potwierdzeniu.",
    ),
    VoiceCommand(
        "recent-activity",
        "co robiłem ostatnio;co robiłam ostatnio;ostatnia aktywność;co się działo na ekranie;pokaz historie ekranu;podsumuj ostatnia aktywnosc;aktywnosc;historia",
        "recent-activity.vbs",
        "Czyta lokalne podsumowanie ostatniej aktywności Screenpipe.",
    ),
    VoiceCommand(
        "active-window",
        "opisz aktywne okno;co mam otwarte;jakie okno jest aktywne;na jakim oknie jestem;podaj aktywne okno;aktywne okno",
        "active-window.vbs",
        "Czyta tytuł aktywnego okna i nazwę programu.",
    ),
    VoiceCommand(
        "check-text-target",
        "gdzie teraz pisze;sprawdz pole tekstowe;czy to pasek adresu;gdzie trafi tekst",
        "check-text-target.vbs",
        "Sprawdza, czy wpisywanie trafi do właściwego pola tekstowego.",
    ),
    VoiceCommand(
        "minimize-window",
        "zminimalizuj okno;schowaj okno;ukryj okno;zwin okno;schowaj aktywne okno;zwin",
        "minimize-window.vbs",
        "Minimalizuje aktualnie aktywne okno.",
    ),
    VoiceCommand(
        "minimize-all",
        "zminimalizuj wszystkie;pokaż pulpit;schowaj wszystkie okna;minimalizuj wszystko;zwin wszystkie okna;pulpit",
        "minimize-all.vbs",
        "Minimalizuje wszystkie okna i pokazuje pulpit.",
    ),
    VoiceCommand(
        "copy-selected-text",
        "kopiuj zaznaczony tekst;skopiuj zaznaczony tekst;kopiuj zaznaczony fragment;przepisz zaznaczenie do schowka;kopiuj zaznaczenie",
        "copy-selected-text.vbs",
        "Kopiuje bieżący zaznaczony tekst do schowka.",
    ),
    VoiceCommand(
        "copy-number-under-cursor",
        "kopiuj numer pod kursorem;skopiuj numer pod kursorem;kopiuj liczbe pod kursorem;skopiuj telefon pod kursorem;kopiuj numer",
        "copy-number-under-cursor.vbs",
        "Kopiuje numer wykryty pod kursorem.",
    ),
    VoiceCommand(
        "copy-sentence-under-cursor",
        "kopiuj cale zdanie pod kursorem;skopiuj cale zdanie pod kursorem;kopiuj tekst pod kursorem;skopiuj zdanie pod myszka;kopiuj zdanie",
        "copy-sentence-under-cursor.vbs",
        "Kopiuje całe zdanie wykryte pod kursorem.",
    ),
    VoiceCommand(
        "listen-start",
        "włącz nasłuch;zacznij nasłuch;wlacz rozmowe;start nasluchu;sluchaj ciagle;nasluch on",
        "listen-start.vbs",
        "Włącza ciągły nasłuch Deepgram.",
    ),
    VoiceCommand(
        "listen-stop",
        "wyłącz nasłuch;zatrzymaj nasłuch;wylacz rozmowe;stop nasluchu;przestan sluchac;nasluch off",
        "listen-stop.vbs",
        "Wyłącza Deepgram bez anulowania wykonywanych akcji.",
    ),
    VoiceCommand(
        "status",
        "status voice loop;czy działasz;jaki jest status;podaj status systemu;jak stoisz;status",
        "status.vbs",
        "Czyta krótki stan rdzenia, modelu i nasłuchu.",
    ),
    VoiceCommand(
        "confirm",
        "potwierdź;tak potwierdzam;wykonaj to;zatwierdz;potwierdz wykonanie;potwierdz",
        "confirm.vbs",
        "Potwierdza najnowsze polecenie oczekujące na zgodę.",
    ),
    VoiceCommand(
        "cancel",
        "anuluj zadanie;nie rób tego;przerwij to;odrzuc polecenie;anuluj wykonanie;anuluj",
        "cancel.vbs",
        "Anuluje najnowsze polecenie oczekujące na zgodę.",
    ),
    VoiceCommand(
        "voice-test",
        "test pętli;test głosu;voice test;test voiceloop;sprawdz glos;test",
        "voice-test.vbs",
        "Sprawdza pełną lokalną pętlę VoiceLoop.",
    ),
    VoiceCommand(
        "open-calendar",
        "otwórz kalendarz;open calendar;uruchom kalendarz;pokaz kalendarz;wlacz kalendarz;kalendarz",
        "open-calendar.vbs",
        "Otwiera kalendarz przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "open-browser",
        "otwórz przeglądarkę;open browser;uruchom przegladarke;otworz internet;odpal przegladarke;przegladarka",
        "open-browser.vbs",
        "Otwiera przeglądarkę przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "search-web",
        "wyszukaj w internecie;sprawdz w necie;szukaj online",
        "search-web.vbs",
        "Uruchamia szybkie wyszukiwanie informacji w internecie.",
    ),
    VoiceCommand(
        "open-chat",
        "otwórz czat;open chat;otworz chatgpt;nowy chat gpt;uruchom chat;chatgpt",
        "open-chat.vbs",
        "Otwiera czat przez dozwoloną akcję.",
    ),
    VoiceCommand(
        "open-gpt-chat",
        "otworz gpt;open gpt;otworz chat gpt;uruchom gpt",
        "open-gpt-chat.vbs",
        "Otwiera osobno czat GPT.",
    ),
    VoiceCommand(
        "open-gemini-chat",
        "otworz gemini;open gemini;czat gemini;uruchom gemini",
        "open-gemini-chat.vbs",
        "Otwiera osobno czat Gemini.",
    ),
    VoiceCommand(
        "stop",
        "stop teraz;przerwij wszystko;natychmiastowy stop;abort;awaryjnie stop;panic stop;stop",
        "stop.vbs",
        "Natychmiast zatrzymuje nasłuch, kolejkę, TTS i bieżącą akcję.",
    ),
)


def set_text(element: ET.Element, name: str, value: str) -> None:
    target = element.find(name)
    if target is None:
        raise RuntimeError(f"Szablon VoiceAttack nie zawiera elementu {name}.")
    target.text = value


def build_profile() -> None:
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
    set_text(root, "Name", "VoiceLoop v2")

    last_command_id = ""
    for definition in COMMANDS:
        command = copy.deepcopy(template_command)
        command_id = str(uuid.uuid5(ID_NAMESPACE, f"command:{definition.key}"))
        action_id = str(uuid.uuid5(ID_NAMESPACE, f"action:{definition.key}"))
        last_command_id = command_id

        set_text(command, "Id", command_id)
        set_text(command, "CommandString", definition.phrases)
        set_text(command, "Description", f"VoiceLoop v2: {definition.description}")
        set_text(command, "Category", "VoiceLoop v2")
        set_text(command, "Async", "true")
        set_text(command, "lastEditedAction", action_id)
        set_text(command, "ExecFromWildcard", "false")

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
    print(f"Zbudowano {OUTPUT_PATH} ({len(COMMANDS)} komend).")


if __name__ == "__main__":
    build_profile()
