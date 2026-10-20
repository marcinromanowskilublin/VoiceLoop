import json
import re
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

from PIL import Image

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


def test_panel_authenticates_private_event_stream() -> None:
    panel = (ROOT / "panel" / "index.html").read_text(encoding="utf-8")

    assert "new EventSource" not in panel
    assert 'fetch("/api/v1/events"' in panel
    assert '"X-VoiceLoop-Token": state.token' in panel


def test_panel_dual_track_demonstrator_is_read_only() -> None:
    panel = (ROOT / "panel" / "index.html").read_text(encoding="utf-8")

    assert "PROTOTYPE / READ-ONLY" in panel
    assert 'api("/api/v1/context/dual-track/status")' in panel
    assert 'api("/api/v1/context/dual-track/traces?limit=20")' in panel
    assert 'api("/api/v1/context/dual-track/status", {' not in panel
    assert 'api("/api/v1/context/dual-track/traces", {' not in panel


def test_context_layer_readme_diagrams_are_accessible_svg() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    diagrams = (
        "voiceloop-dual-track-flow.svg",
        "voiceloop-arbiter-outcomes.svg",
        "voiceloop-time-certainty.svg",
    )
    first_section = readme.index("## Context first")

    for name in diagrams:
        relative = f"docs/img/{name}"
        image = ROOT / relative
        root = ET.parse(image).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        title = root.find("svg:title", namespace)
        description = root.find("svg:desc", namespace)
        markdown = re.search(
            rf"!\[([^\]]+)\]\({re.escape(relative)}\)",
            readme,
        )

        assert root.attrib["role"] == "img"
        assert title is not None and len(title.text or "") >= 20
        assert description is not None and len(description.text or "") >= 80
        assert markdown is not None and len(markdown.group(1)) >= 60
        assert readme.index(relative) < first_section


def test_context_layer_readme_treats_voice_as_one_adapter() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    introduction = readme[: readme.index("## Safety Model")]
    normalized = " ".join(introduction.split())

    assert "A local-first context layer" in normalized
    assert "not primarily a voice assistant" in normalized
    assert "| Voice | optional STT or VoiceAttack input adapter |" in introduction
    assert "Voice can submit the same bounded input as text" in normalized
    assert "100% retrieval accuracy" in normalized


def test_updated_documentation_has_no_missing_local_links() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "DO_WYKONANIA_PRZEZ_CZLOWIEKA.md",
    )
    for document in documents:
        source = document.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (document.parent / target).resolve().exists(), (
                f"{document.relative_to(ROOT)} -> {raw_target}"
            )


def test_public_context_evidence_is_committed_and_ci_generated() -> None:
    evidence = ROOT / "docs" / "evaluation" / "context-evidence-v1"
    report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
    markdown = (evidence / "report.md").read_text(encoding="utf-8")
    html = (evidence / "report.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert report["source_revision"] == "372b642"
    assert report["network_used"] is False
    assert report["storage"]["canonical"] == "SQLite"
    assert report["storage"]["derived_index"] == "Qdrant local mode"
    assert report["ingest"]["idempotent"] is True
    assert report["evaluation"]["promotion_allowed"] is False
    assert {item["name"] for item in report["scenarios"]} == {
        "agreement",
        "conflict",
        "abstention",
        "unbounded",
    }
    assert "**rrf5:**" in markdown.casefold()
    assert "topic:miss" in markdown.casefold()
    assert "PUBLIC SYNTHETIC EVIDENCE" in html
    assert "scripts\\run_context_evidence_demo.py" in readme
    assert "docs/evaluation/context-evidence-v1/report.md" in readme
    assert "run_context_evidence_demo.py" in workflow
    assert "actions/upload-artifact@v6" in workflow

    with Image.open(ROOT / "docs" / "img" / "voiceloop-context-evidence.png") as image:
        assert image.size == (1440, 1200)


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
    assert "Start session" in panel
    assert "/api/v1/conversation/start" in panel
    assert "Gemini is the primary model" in panel or "geminiPrimary" in panel
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
    assert all(
        (ROOT / "scripts" / "va" / PureWindowsPath(path).name).is_file()
        for path in paths
        if path
    )


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
        PureWindowsPath(path)
        for action in profile.findall(".//CommandAction")
        if action.findtext("ActionType") == "Launch"
        if (path := action.findtext("Context"))
    ]

    assert profile.findtext(".//Name") == "VoiceLoop v2 PRO"
    assert len(commands) == 34
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
    assert any(phrase.startswith("kursor;") for phrase in phrases)
    assert all(
        action.findtext("ActionType") == "Launch"
        for action in profile.findall(".//CommandAction")
    )
    assert all(path.parent.name.casefold() == "va" for path in paths)
    assert all(
        (ROOT / "scripts" / "va" / path.name).is_file() and path.suffix == ".vbs"
        for path in paths
    )


def test_voiceattack_action_audit_validates_profile_without_execution() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts" / "check_voiceattack_actions.py"))

    report = namespace["audit_voiceattack_actions"](live=False)

    assert report["safe_mode"] is True
    assert report["status"] == "ok"
    assert report["errors"] == []
    assert report["profile"]["name"] == "VoiceLoop v2 PRO"
    assert report["profile"]["command_count"] == 34
    assert report["profile"]["phrase_count"] == 655
    assert report["profile"]["missing_scripts"] == 0
    assert len(report["inventory"]) == 34


def test_one_click_launcher_starts_safe_complete_stack() -> None:
    launcher = (ROOT / "Start-VoiceLoop.bat").read_text(encoding="utf-8")
    start_all = (ROOT / "scripts" / "start-all.ps1").read_text(encoding="utf-8")
    qdrant = (ROOT / "scripts" / "start-qdrant.ps1").read_text(encoding="utf-8")
    screenpipe = (ROOT / "scripts" / "start-screenpipe.ps1").read_text(
        encoding="utf-8"
    )

    assert "scripts\\start-all.ps1" in launcher
    assert "& $startQdrant" in start_all
    assert "check_voiceattack_actions.py" in start_all
    assert "VoiceAttack" in start_all
    assert "-ContextOnly" in start_all
    assert "FullScreenpipeCapture" in start_all
    assert "qdrant/qdrant:v1.19.0" in qdrant
    assert "docker update --restart $restartPolicy" in qdrant
    assert "--restart $restartPolicy" in qdrant
    assert "--health-cmd" in qdrant
    assert "--log-opt 'max-size=10m'" in qdrant
    assert "--disable-audio" in screenpipe
    assert "--disable-keyboard-capture" in screenpipe
    assert "--disable-clipboard-capture" in screenpipe
    assert "--disable-click-capture" in screenpipe


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
