import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voiceloop import windows_shell as windows_shell_module
from voiceloop.windows_shell import (
    ShellItem,
    WindowsShellLocator,
    center_cursor,
    filter_by_extension,
    filter_by_first_letter,
    get_cursor_position,
    hover_shell_item,
    layout_rect,
    locate_active_folder_handle,
    match_active_folder_item,
    match_shell_item,
    monitor_work_area_at,
    monitor_work_area_for_window,
    open_shell_item,
    revalidate_active_folder_item,
    select_shell_items,
    set_cursor_position,
)


def item(name: str, runtime: int = 1) -> ShellItem:
    return ShellItem(name, (10, 20, 110, 120), "ListItem", "Progman", 7, (runtime,))


def folder_item(name: str, runtime: int = 1, handle: int = 42) -> ShellItem:
    return ShellItem(
        name,
        (10, 20, 110, 120),
        "ListItem",
        "CabinetWClass",
        handle,
        (runtime,),
        is_folder=True,
        extension="",
    )


def file_item(
    name: str,
    runtime: int = 1,
    handle: int = 42,
    extension: str = "",
) -> ShellItem:
    return ShellItem(
        name,
        (10, 20, 110, 120),
        "ListItem",
        "CabinetWClass",
        handle,
        (runtime,),
        is_folder=False,
        extension=extension,
    )


def test_exact_and_polish_stt_matching() -> None:
    candidates = [item("Mortal Shell"), item("A Way Out", 2)]
    assert match_shell_item("mortal shell", candidates).name == "Mortal Shell"
    assert match_shell_item("Mortal Szel", candidates).name == "Mortal Shell"
    assert match_shell_item("A łej aut", candidates).name == "A Way Out"


def test_match_shell_item_does_not_crash_on_tied_zero_scores() -> None:
    # Regresja: wśród wielu niepowiązanych elementów (np. cała lista ikon
    # pulpitu) często kilka par ma identyczny wynik podobieństwa (0.0), a
    # ShellItem nie jest porównywalny operatorem "<". sorted() nie może więc
    # sortować całych krotek (score, item) i musi sortować tylko po score.
    candidates = [item("Zupelnie inna nazwa", runtime=1), item("Jeszcze inna", runtime=2)]

    with pytest.raises(RuntimeError, match="Nie znalazłem"):
        match_shell_item("Mortal Shell", candidates)


def test_match_rejects_zero_and_two_equal_hits() -> None:
    with pytest.raises(RuntimeError, match="Nie znalazłem"):
        match_shell_item("Nie istnieje", [item("Mortal Shell")])
    with pytest.raises(RuntimeError, match="kilka równorzędnych"):
        match_shell_item("Mortal Shell", [item("Mortal Shell"), item("Mortal Shell", 2)])


def test_hover_only_sets_cursor_position(monkeypatch) -> None:
    target = item("A Way Out")
    monkeypatch.setattr(
        WindowsShellLocator,
        "validate",
        lambda self, expected: (target, Mock()),
    )
    win32api = SimpleNamespace(SetCursorPos=Mock())
    monkeypatch.setitem(__import__("sys").modules, "win32api", win32api)

    result = hover_shell_item(target.identity())

    assert result == target
    win32api.SetCursorPos.assert_called_once_with((60, 70))
    assert not hasattr(win32api, "mouse_event")


def test_open_revalidates_and_invokes_uia(monkeypatch) -> None:
    target = item("Mortal Shell")
    wrapper = Mock()
    validate = Mock(return_value=(target, wrapper))
    monkeypatch.setattr(WindowsShellLocator, "validate", validate)

    result = open_shell_item(target.identity())

    assert result == target
    validate.assert_called_once_with(target.identity())
    wrapper.invoke.assert_called_once_with()
    wrapper.type_keys.assert_not_called()


# ---------------------------------------------------------------------------
# "Aktywny folder" = okno Eksploratora pod kursorem myszy (bez fallbacku).
# ---------------------------------------------------------------------------


def test_locate_active_folder_handle_returns_explorer_root(monkeypatch) -> None:
    win32api = SimpleNamespace(GetCursorPos=Mock(return_value=(5, 6)))
    win32con = SimpleNamespace(GA_ROOT=2)
    win32gui = SimpleNamespace(
        WindowFromPoint=Mock(return_value=99),
        GetAncestor=Mock(return_value=42),
        GetClassName=Mock(return_value="CabinetWClass"),
    )
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)

    handle = locate_active_folder_handle()

    assert handle == 42
    win32gui.GetAncestor.assert_called_once_with(99, 2)


def test_locate_active_folder_handle_rejects_non_explorer_window(monkeypatch) -> None:
    win32api = SimpleNamespace(GetCursorPos=Mock(return_value=(5, 6)))
    win32con = SimpleNamespace(GA_ROOT=2)
    win32gui = SimpleNamespace(
        WindowFromPoint=Mock(return_value=99),
        GetAncestor=Mock(return_value=42),
        GetClassName=Mock(return_value="Progman"),
    )
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)

    with pytest.raises(RuntimeError, match="nie ma okna Eksploratora"):
        locate_active_folder_handle()


def test_locate_active_folder_handle_rejects_no_window_under_cursor(monkeypatch) -> None:
    win32api = SimpleNamespace(GetCursorPos=Mock(return_value=(5, 6)))
    win32con = SimpleNamespace(GA_ROOT=2)
    win32gui = SimpleNamespace(WindowFromPoint=Mock(return_value=0))
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)

    with pytest.raises(RuntimeError, match="Nie znalazłem okna pod kursorem"):
        locate_active_folder_handle()


def test_enumerate_active_folder_items_scopes_to_single_root(monkeypatch) -> None:
    def make_wrapper(name: str, runtime: int, control_type: str = "ListItem") -> Mock:
        wrapper = Mock()
        wrapper.is_visible.return_value = True
        wrapper.element_info = SimpleNamespace(
            name=name,
            control_type=control_type,
            runtime_id=(runtime,),
            automation_id="",
        )
        wrapper.rectangle.return_value = SimpleNamespace(left=0, top=0, right=50, bottom=20)
        return wrapper

    root = Mock()
    root.is_visible.return_value = True
    root.descendants.return_value = [make_wrapper("Faktura.pdf", 1)]

    desktop_instance = Mock()
    desktop_instance.window.return_value = root
    pywinauto_stub = SimpleNamespace(Desktop=Mock(return_value=desktop_instance))
    win32gui = SimpleNamespace(GetClassName=Mock(return_value="CabinetWClass"))
    monkeypatch.setitem(sys.modules, "pywinauto", pywinauto_stub)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setattr(
        windows_shell_module,
        "_attach_folder_metadata",
        lambda handle, items: items,
    )

    items = WindowsShellLocator().enumerate_active_folder_items(42)

    assert [entry.name for entry in items] == ["Faktura.pdf"]
    assert items[0].root_handle == 42
    assert items[0].root_class == "CabinetWClass"
    desktop_instance.window.assert_called_once_with(handle=42)


def test_enumerate_active_folder_items_rejects_non_explorer_handle(monkeypatch) -> None:
    win32gui = SimpleNamespace(GetClassName=Mock(return_value="Progman"))
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Desktop=Mock()))

    with pytest.raises(RuntimeError, match="nie jest Eksploratorem"):
        WindowsShellLocator().enumerate_active_folder_items(42)


def test_revalidate_active_folder_item_detects_change(monkeypatch) -> None:
    original = folder_item("Projekty", runtime=1)
    monkeypatch.setattr(
        WindowsShellLocator,
        "enumerate_active_folder_items",
        lambda self, handle: [],
    )

    with pytest.raises(RuntimeError, match="zmienił się lub zniknął"):
        revalidate_active_folder_item(42, original)


def test_revalidate_active_folder_item_returns_fresh_copy(monkeypatch) -> None:
    original = folder_item("Projekty", runtime=1)
    fresh = folder_item("Projekty", runtime=1)
    monkeypatch.setattr(
        WindowsShellLocator,
        "enumerate_active_folder_items",
        lambda self, handle: [fresh],
    )

    result = revalidate_active_folder_item(42, original)

    assert result is fresh


# ---------------------------------------------------------------------------
# Exact / fuzzy matching z 2-3 kandydatami dla niejednoznacznych nazw.
# ---------------------------------------------------------------------------


def test_match_active_folder_item_exact_match_wins() -> None:
    candidates = [file_item("Raport"), file_item("Raport Q2", runtime=2)]

    match, alternatives = match_active_folder_item("Raport", candidates)

    assert match is not None
    assert match.name == "Raport"
    assert alternatives == []


def test_match_active_folder_item_returns_two_to_three_candidates() -> None:
    candidates = [
        file_item("Raport styczen", runtime=1),
        file_item("Raport luty", runtime=2),
        file_item("Raport marzec", runtime=3),
    ]

    match, alternatives = match_active_folder_item("Raport", candidates)

    assert match is None
    assert 2 <= len(alternatives) <= 3


def test_match_active_folder_item_never_autoselects_ambiguous() -> None:
    candidates = [
        file_item("Raport styczen", runtime=1),
        file_item("Raport luty", runtime=2),
    ]

    match, alternatives = match_active_folder_item("Raport", candidates)

    assert match is None
    assert all(candidate in candidates for candidate in alternatives)


def test_match_active_folder_item_raises_when_nothing_plausible() -> None:
    with pytest.raises(RuntimeError, match="Nie znalazłem"):
        match_active_folder_item("Zupelnie inna nazwa", [file_item("Raport")])


def test_match_active_folder_item_filters_by_kind() -> None:
    candidates = [folder_item("Atak"), file_item("Atak.pdf", runtime=2)]

    match, _ = match_active_folder_item("Atak", candidates, kind="folder")

    assert match is not None
    assert match.is_folder is True


def test_match_active_folder_item_kind_filter_raises_without_metadata() -> None:
    candidates = [item("Nieznany typ")]

    with pytest.raises(RuntimeError, match="rozróżnić plików i folderów"):
        match_active_folder_item("Nieznany typ", candidates, kind="file")


# ---------------------------------------------------------------------------
# Filtry PDF / pierwsza litera.
# ---------------------------------------------------------------------------


def test_filter_by_extension_matches_case_insensitively() -> None:
    candidates = [
        file_item("Raport.pdf", runtime=1, extension="pdf"),
        file_item("Zdjecie.jpg", runtime=2, extension="jpg"),
    ]

    result = filter_by_extension(candidates, "PDF")

    assert [entry.name for entry in result] == ["Raport.pdf"]


def test_filter_by_first_letter_matches_case_insensitively() -> None:
    candidates = [
        file_item("Atak.pdf", runtime=1, extension="pdf"),
        file_item("Budzet.xlsx", runtime=2, extension="xlsx"),
    ]

    result = filter_by_first_letter(candidates, "a")

    assert [entry.name for entry in result] == ["Atak.pdf"]


# ---------------------------------------------------------------------------
# Wielokrotne zaznaczanie wyłącznie przez UIA SelectionItem.
# ---------------------------------------------------------------------------


def test_select_shell_items_uses_selection_item_pattern(monkeypatch) -> None:
    first = folder_item("Atak.pdf", runtime=1)
    second = folder_item("Atak2.pdf", runtime=2)

    def make_wrapper(runtime_id: tuple[int, ...]) -> Mock:
        wrapper = Mock()
        wrapper.element_info = SimpleNamespace(runtime_id=runtime_id)
        wrapper.iface_selection_item = Mock()
        return wrapper

    wrapper1 = make_wrapper((1,))
    wrapper2 = make_wrapper((2,))
    root = Mock()
    root.descendants.return_value = [wrapper1, wrapper2]
    desktop_instance = Mock()
    desktop_instance.window.return_value = root
    monkeypatch.setitem(
        sys.modules,
        "pywinauto",
        SimpleNamespace(Desktop=Mock(return_value=desktop_instance)),
    )

    selected = select_shell_items(42, [first, second])

    assert selected == [first, second]
    wrapper1.iface_selection_item.Select.assert_called_once_with()
    wrapper1.iface_selection_item.AddToSelection.assert_not_called()
    wrapper2.iface_selection_item.AddToSelection.assert_called_once_with()
    wrapper2.iface_selection_item.Select.assert_not_called()


def test_select_shell_items_rejects_missing_target(monkeypatch) -> None:
    target = folder_item("Zniknal", runtime=5)
    root = Mock()
    root.descendants.return_value = []
    desktop_instance = Mock()
    desktop_instance.window.return_value = root
    monkeypatch.setitem(
        sys.modules,
        "pywinauto",
        SimpleNamespace(Desktop=Mock(return_value=desktop_instance)),
    )

    with pytest.raises(RuntimeError, match="zmienił się lub zniknął"):
        select_shell_items(42, [target])


# ---------------------------------------------------------------------------
# Powrót i środek kursora.
# ---------------------------------------------------------------------------


def test_get_and_set_cursor_position(monkeypatch) -> None:
    win32api = SimpleNamespace(
        GetCursorPos=Mock(return_value=(11, 22)),
        SetCursorPos=Mock(),
    )
    monkeypatch.setitem(sys.modules, "win32api", win32api)

    assert get_cursor_position() == (11, 22)
    set_cursor_position((33, 44))
    win32api.SetCursorPos.assert_called_once_with((33, 44))


def test_center_cursor_uses_monitor_work_area(monkeypatch) -> None:
    win32api = SimpleNamespace(
        GetCursorPos=Mock(return_value=(11, 22)),
        SetCursorPos=Mock(),
        MonitorFromPoint=Mock(return_value="monitor-handle"),
        GetMonitorInfo=Mock(return_value={"Work": (0, 0, 1920, 1080)}),
    )
    win32con = SimpleNamespace(MONITOR_DEFAULTTONEAREST=2)
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)

    center = center_cursor()

    assert center == (960, 540)
    win32api.SetCursorPos.assert_called_once_with((960, 540))


def test_monitor_work_area_for_window(monkeypatch) -> None:
    win32api = SimpleNamespace(
        MonitorFromWindow=Mock(return_value="monitor-handle"),
        GetMonitorInfo=Mock(return_value={"Work": (0, 0, 2560, 1440)}),
    )
    win32con = SimpleNamespace(MONITOR_DEFAULTTONEAREST=2)
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)

    assert monitor_work_area_for_window(123) == (0, 0, 2560, 1440)


def test_monitor_work_area_at(monkeypatch) -> None:
    win32api = SimpleNamespace(
        MonitorFromPoint=Mock(return_value="monitor-handle"),
        GetMonitorInfo=Mock(return_value={"Work": (0, 0, 1920, 1080)}),
    )
    win32con = SimpleNamespace(MONITOR_DEFAULTTONEAREST=2)
    monkeypatch.setitem(sys.modules, "win32api", win32api)
    monkeypatch.setitem(sys.modules, "win32con", win32con)

    assert monitor_work_area_at((10, 10)) == (0, 0, 1920, 1080)


# ---------------------------------------------------------------------------
# Układy okien liczone względem obszaru roboczego monitora.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("left_half", (0, 0, 960, 1080)),
        ("right_half", (960, 0, 960, 1080)),
        ("left_third", (0, 0, 640, 1080)),
        ("center_third", (640, 0, 640, 1080)),
        ("right_third", (1280, 0, 640, 1080)),
        ("top_left_quarter", (0, 0, 960, 540)),
        ("bottom_left_quarter", (0, 540, 960, 540)),
        ("top_right_quarter", (960, 0, 960, 540)),
        ("bottom_right_quarter", (960, 540, 960, 540)),
    ],
)
def test_layout_rect_covers_all_nine_layouts(layout, expected) -> None:
    work_area = (0, 0, 1920, 1080)

    assert layout_rect(work_area, layout) == expected


def test_layout_rect_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="Nieznany układ okna"):
        layout_rect((0, 0, 1920, 1080), "diagonal_split")


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("left_half", (100, 50, 860, 900)),
        ("right_half", (960, 50, 860, 900)),
    ],
)
def test_layout_rect_respects_secondary_monitor_offset(layout, expected) -> None:
    # Symuluje drugi monitor przesunięty w prawo/dół względem (0, 0).
    work_area = (100, 50, 1820, 950)

    assert layout_rect(work_area, layout) == expected
