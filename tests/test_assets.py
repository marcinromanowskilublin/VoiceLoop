import json
import xml.etree.ElementTree as ET
from pathlib import Path

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
        if action.findtext("ActionType") == "RunApplication"
    ]

    assert {"voice test", "open calendar", "open browser", "open chat"}.issubset(command_names)
    assert all(path and path.lower().endswith(".vbs") for path in paths)
    assert all(Path(path).exists() for path in paths if path)


def test_note_macro_uses_runtime_text_argument() -> None:
    macro = json.loads(
        (ROOT / "uivision" / "macros" / "voiceloop_notatka.json").read_text(encoding="utf-8")
    )
    type_commands = [command for command in macro["Commands"] if command["Command"] == "XType"]

    assert type_commands[0]["Target"] == "${!cmd_var1}"
