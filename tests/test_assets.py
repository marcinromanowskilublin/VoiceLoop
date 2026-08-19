import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from voiceloop.models import CommandRequest, CommandSource
from voiceloop.router import deterministic_plan

ROOT = Path(__file__).resolve().parents[1]


def test_n8n_workflow_has_no_execute_command() -> None:
    workflow = json.loads((ROOT / "n8n" / "voice-loop.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in workflow["nodes"]}

    assert "n8n-nodes-base.executeCommand" not in node_types
    assert any(node["type"] == "n8n-nodes-base.webhook" for node in workflow["nodes"])


def test_panel_does_not_embed_deepgram_key() -> None:
    panels = [
        (ROOT / "panel" / "index.html").read_text(encoding="utf-8"),
        (ROOT / "panel" / "deepgram.html").read_text(encoding="utf-8"),
    ]

    assert "DEFAULT_KEY" not in "\n".join(panels)
    assert "DEEPGRAM_API_KEY=" not in "\n".join(panels)


def test_legacy_deepgram_entrypoints_are_retired() -> None:
    panel = (ROOT / "panel" / "deepgram.html").read_text(encoding="utf-8")
    listener = (ROOT / "listener" / "deepgram_listener.py").read_text(encoding="utf-8")

    assert 'window.location.replace("/")' in panel
    assert "/webhook/voice" not in panel
    assert "/webhook/voice" not in listener
    assert "wycofany" in listener


def test_panel_explains_cloud_primary_mode() -> None:
    panel = (ROOT / "panel" / "index.html").read_text(encoding="utf-8")
    env_example = (ROOT / "listener" / ".env.example").read_text(encoding="utf-8")

    assert "llm_primary" in panel
    assert "Rozpocznij rozmowę" in panel
    assert "/api/v1/conversation/start" in panel
    assert "Gemini jest modelem głównym" in panel or "geminiPrimary" in panel
    assert "GEMINI_MODEL=gemini-3.6-flash" in env_example
    assert "AUTO_START_CONVERSATION=false" in env_example
    assert "Model lokalny jest używany domyślnie" not in panel


def test_voiceattack_profile_points_to_existing_vbs_files() -> None:
    profile = ET.parse(ROOT / "voiceattack" / "VoiceLoop-profil.vap")
    commands = profile.findall(".//Command")
    command_names = {
        command.findtext("CommandString")
        for command in commands
        if command.findtext("CommandString")
    }
    paths = [
        action.findtext("Context")
        for action in profile.findall(".//CommandAction")
        if action.findtext("ActionType") == "Launch"
    ]

    assert {"voice test", "open calendar", "open browser", "open chat"}.issubset(command_names)
    assert all(
        action.findtext("ActionType") == "Launch"
        for action in profile.findall(".//CommandAction")
    )
    assert all(path and path.lower().endswith(".vbs") for path in paths)
    assert all(Path(path).exists() for path in paths if path)


def test_voiceattack_v2_profile_contains_safe_polish_package() -> None:
    profile = ET.parse(ROOT / "voiceattack" / "VoiceLoop-v2.vap")
    commands = profile.findall(".//Command")
    command_ids = [command.findtext("Id") for command in commands]
    action_ids = [action.findtext("Id") for action in profile.findall(".//CommandAction")]
    phrases = {command.findtext("CommandString") or "" for command in commands}
    individual_phrases = [
        phrase.strip().casefold()
        for command_phrases in phrases
        for phrase in command_phrases.split(";")
        if phrase.strip()
    ]
    paths = [
        Path(path)
        for action in profile.findall(".//CommandAction")
        if action.findtext("ActionType") == "Launch"
        if (path := action.findtext("Context"))
    ]

    assert profile.findtext(".//Name") == "VoiceLoop v2 PRO"
    assert len(commands) == 33
    assert len(individual_phrases) >= 620
    assert len(individual_phrases) == len(set(individual_phrases))
    assert len(command_ids) == len(set(command_ids))
    assert len(action_ids) == len(set(action_ids))
    assert {command.findtext("minimumConfidenceLevel") for command in commands} == {"65"}
    assert {command.findtext("UseConfidence") for command in commands} == {"true"}
    assert any(phrase.startswith("asystent;") for phrase in phrases)
    assert any("zminimalizuj okno" in phrase for phrase in phrases)
    assert any("zminimalizuj aplikację pod kursorem" in phrase for phrase in phrases)
    assert any("pokaż pulpit" in phrase for phrase in phrases)
    assert any("wyłącz aplikację pod kursorem" in phrase for phrase in phrases)
    assert any("kopiuj tekst pod kursorem" in phrase for phrase in phrases)
    assert any("skopiuj mail pod myszką" in phrase for phrase in phrases)
    assert any("zaznacz zdanie pod kursorem" in phrase for phrase in phrases)
    assert any("zaznacz akapit pod kursorem" in phrase for phrase in phrases)
    assert any("zapamiętaj ostatnie źródło" in phrase for phrase in phrases)
    assert any("natychmiastowy stop" in phrase for phrase in phrases)
    assert all(
        action.findtext("ActionType") == "Launch"
        for action in profile.findall(".//CommandAction")
    )
    assert all(path.parent == ROOT / "scripts" / "va" for path in paths)
    assert all(path.is_file() and path.suffix == ".vbs" for path in paths)


def test_voiceattack_dispatcher_preserves_fixed_command_ids() -> None:
    dispatcher = (ROOT / "scripts" / "send-command.ps1").read_text(encoding="utf-8")

    assert "Resolve-AutoRouteText" not in dispatcher
    assert "$conversationMode" not in dispatcher
    assert "$CommandId = $null" not in dispatcher
    assert "command_id = if ($CommandId) { $CommandId } else { $null }" in dispatcher


def test_voiceattack_fixed_wrappers_have_deterministic_fast_paths() -> None:
    wrappers = sorted((ROOT / "scripts" / "va").glob("*.vbs"))
    command_ids: dict[str, str] = {}

    for wrapper in wrappers:
        source = wrapper.read_text(encoding="utf-8")
        match = re.search(r"-CommandId\s+([A-Za-z0-9_]+)", source)
        if match is None:
            continue
        command_id = match.group(1)
        command_ids[wrapper.name] = command_id
        plan = deterministic_plan(
            CommandRequest(
                source=CommandSource.VOICEATTACK,
                command_id=command_id,
            )
        )
        assert plan is not None, f"{wrapper.name}: brak szybkiej ścieżki dla {command_id}"

    assert command_ids


def test_note_macro_uses_runtime_text_argument() -> None:
    macro = json.loads(
        (ROOT / "uivision" / "macros" / "voiceloop_notatka.json").read_text(encoding="utf-8")
    )
    type_commands = [command for command in macro["Commands"] if command["Command"] == "XType"]

    assert type_commands[0]["Target"] == "${!cmd_var1}"
