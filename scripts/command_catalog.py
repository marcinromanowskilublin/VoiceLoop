"""Thirty grounded VoiceLoop Windows calibration phrases, no action execution."""

TARGET_PER_COMMAND = 30

COMMANDS = (
    {"id": "minimalizuj", "phrase": "Minimalizuj", "action_id": "minimize_all_windows", "category": "Okna", "detail": "Wszystkie okna"},
    {"id": "pulpit", "phrase": "Pulpit", "action_id": "minimize_all_windows", "category": "Okna", "detail": "Pokazuje pulpit"},
    {"id": "zwin", "phrase": "Zwiń", "action_id": "minimize_active_window", "category": "Okna", "detail": "Aktywne okno"},
    {"id": "zwin-pod-kursorem", "phrase": "Zwiń okno pod kursorem", "action_id": "minimize_window_under_cursor", "category": "Okna", "detail": "Wskazane okno"},
    {"id": "zamknij-pod-kursorem", "phrase": "Zamknij okno pod kursorem", "action_id": "close_window_under_cursor", "category": "Okna", "detail": "Wymaga potwierdzenia"},
    {"id": "otworz-chrome", "phrase": "Otwórz Chrome", "action_id": "open_browser", "category": "Otwieranie", "detail": "Przeglądarka"},
    {"id": "otworz-kalendarz", "phrase": "Otwórz kalendarz", "action_id": "open_calendar", "category": "Otwieranie", "detail": "Kalendarz"},
    {"id": "otworz-whatsapp", "phrase": "Otwórz WhatsApp", "action_id": "open_app", "category": "Otwieranie", "detail": "WhatsApp"},
    {"id": "otworz-komputer", "phrase": "Otwórz Ten komputer", "action_id": "open_folder", "category": "Otwieranie", "detail": "Eksplorator Windows"},
    {"id": "otworz-chatgpt", "phrase": "Otwórz ChatGPT", "action_id": "open_chat", "category": "Otwieranie", "detail": "ChatGPT"},
    {"id": "otworz-gemini", "phrase": "Otwórz Gemini", "action_id": "open_gemini_chat", "category": "Otwieranie", "detail": "Gemini"},
    {"id": "otworz-czat", "phrase": "Otwórz czat", "action_id": "open_chat", "category": "Otwieranie", "detail": "Czat"},
    {"id": "aktywne-okno", "phrase": "Jakie okno jest aktywne?", "action_id": "describe_active_window", "category": "Pytania", "detail": "Aktualne okno"},
    {"id": "ostatnia-aktywnosc", "phrase": "Co ostatnio robiłem?", "action_id": "describe_recent_activity", "category": "Pytania", "detail": "Historia aktywności"},
    {"id": "kursor-srodek", "phrase": "Kursor na środek", "action_id": "cursor_center", "category": "Kursor", "detail": "Środek ekranu"},
    {"id": "wroc-kursorem", "phrase": "Wróć kursorem", "action_id": "cursor_return", "category": "Kursor", "detail": "Poprzednia pozycja"},
    {"id": "okno-lewa", "phrase": "Przesuń okno na lewą połowę", "action_id": "snap_window_layout", "category": "Okna", "detail": "Lewa połowa"},
    {"id": "okno-prawa", "phrase": "Przesuń okno na prawą połowę", "action_id": "snap_window_layout", "category": "Okna", "detail": "Prawa połowa"},
    {"id": "kopiuj-zaznaczenie", "phrase": "Kopiuj zaznaczenie", "action_id": "copy_selected_text", "category": "Tekst", "detail": "Schowek"},
    {"id": "kopiuj-email", "phrase": "Kopiuj e-mail pod kursorem", "action_id": "copy_email_under_cursor", "category": "Tekst", "detail": "Adres e-mail"},
    {"id": "kopiuj-numer", "phrase": "Kopiuj numer pod kursorem", "action_id": "copy_number_under_cursor", "category": "Tekst", "detail": "Numer"},
    {"id": "kopiuj-tekst", "phrase": "Kopiuj tekst pod kursorem", "action_id": "copy_text_under_cursor", "category": "Tekst", "detail": "Tekst"},
    {"id": "kopiuj-zdanie", "phrase": "Kopiuj zdanie pod kursorem", "action_id": "copy_sentence_under_cursor", "category": "Tekst", "detail": "Zdanie"},
    {"id": "zaznacz-zdanie", "phrase": "Zaznacz zdanie", "action_id": "select_sentence_under_cursor", "category": "Tekst", "detail": "Zdanie pod kursorem"},
    {"id": "zaznacz-akapit", "phrase": "Zaznacz akapit", "action_id": "select_paragraph_under_cursor", "category": "Tekst", "detail": "Akapit pod kursorem"},
    {"id": "odczytaj-notatke", "phrase": "Odczytaj notatkę", "action_id": "read_active_notepad", "category": "Tekst", "detail": "Aktywny Notatnik"},
    {"id": "gdzie-pisze", "phrase": "Gdzie piszę?", "action_id": "describe_text_target", "category": "Pytania", "detail": "Pole tekstowe"},
    {"id": "co-potrafisz", "phrase": "Co potrafisz?", "action_id": "list_capabilities", "category": "Pytania", "detail": "Lista możliwości"},
    {"id": "zaznacz-a", "phrase": "Zaznacz wszystkie na literę A", "action_id": "select_shell_items_by_letter", "category": "Pliki", "detail": "Eksplorator"},
    {"id": "stop", "phrase": "Stop", "action_id": "stop", "category": "Sterowanie", "detail": "Przerwanie zadania"},
)

COMMAND_BY_ID = {item["id"]: item for item in COMMANDS}
