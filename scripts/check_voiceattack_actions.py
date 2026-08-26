"""Bezpieczny audyt profilu VoiceAttack — bez wykonywania komend.

Sprawdza cztery warstwy:
1. definicje generatora profilu,
2. wygenerowany plik ``VoiceLoop-v2.vap``,
3. skrypty transportowe ``scripts/va/*.vbs``,
4. opcjonalnie żywy katalog możliwości VoiceLoop.

Audyt nie uruchamia VoiceAttack, nie importuje profilu i nie wywołuje żadnej
akcji systemowej. Raport domyślnie trafia do
``logs/voiceattack-action-audit.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "voiceattack" / "VoiceLoop-v2.vap"
GENERATOR_PATH = ROOT / "scripts" / "build-voiceattack-profile.py"
SCRIPTS_DIR = ROOT / "scripts" / "va"
TOKEN_PATH = ROOT / "data" / "voiceloop.token"
DEFAULT_REPORT_PATH = ROOT / "logs" / "voiceattack-action-audit.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8765/api/v1"

COMMAND_ID_PATTERN = re.compile(r"-CommandId\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
OPERATION_PATTERN = re.compile(r"-Operation\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
ACTION_ALIASES = {
    "active_window": "describe_active_window",
    "recent_activity": "describe_recent_activity",
}


def _normalise_phrase(value: str) -> str:
    return " ".join(value.casefold().split())


def _voiceattack_running() -> bool:
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            [
                "tasklist",
                "/FI",
                "IMAGENAME eq VoiceAttack.exe",
                "/FO",
                "CSV",
                "/NH",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return '"VoiceAttack.exe"' in result.stdout


def _load_generator_commands() -> list[Any]:
    namespace = runpy.run_path(str(GENERATOR_PATH))
    commands = namespace.get("COMMANDS")
    if not isinstance(commands, tuple | list):
        raise TypeError("Generator nie udostępnia listy COMMANDS.")
    return list(commands)


def _read_wrapper(path: Path) -> dict[str, str | None]:
    source = path.read_text(encoding="utf-8", errors="replace")
    command_match = COMMAND_ID_PATTERN.search(source)
    operation_match = OPERATION_PATTERN.search(source)
    return {
        "command_id": command_match.group(1) if command_match else None,
        "operation": operation_match.group(1) if operation_match else None,
    }


def _fetch_capabilities(base_url: str, timeout: float) -> dict[str, Any]:
    if not TOKEN_PATH.is_file():
        raise RuntimeError(f"Brak lokalnego tokenu: {TOKEN_PATH}")
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/capabilities",
        headers={"X-VoiceLoop-Token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Nie udało się odczytać katalogu VoiceLoop: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Katalog VoiceLoop nie jest obiektem JSON.")
    return payload


def audit_voiceattack_actions(
    *,
    live: bool = True,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    inventory: list[dict[str, Any]] = []

    if not GENERATOR_PATH.is_file():
        errors.append(f"Brak generatora profilu: {GENERATOR_PATH}")
        commands: list[Any] = []
    else:
        try:
            commands = _load_generator_commands()
        except Exception as exc:  # noqa: BLE001 - raport ma zebrać wszystkie problemy
            errors.append(str(exc))
            commands = []

    profile_commands: list[ET.Element] = []
    profile_name = ""
    if not PROFILE_PATH.is_file():
        errors.append(f"Brak profilu: {PROFILE_PATH}")
    else:
        try:
            root = ET.parse(PROFILE_PATH).getroot()
            profile_name = str(root.findtext("Name") or "")
            commands_node = root.find("Commands")
            profile_commands = list(commands_node) if commands_node is not None else []
        except (OSError, ET.ParseError) as exc:
            errors.append(f"Nieprawidłowy profil VoiceAttack: {exc}")

    if profile_name and profile_name != "VoiceLoop v2 PRO":
        errors.append(f"Nieoczekiwana nazwa profilu: {profile_name!r}")
    if commands and profile_commands and len(commands) != len(profile_commands):
        errors.append(
            f"Generator ma {len(commands)} komend, a profil {len(profile_commands)}."
        )

    phrase_owners: dict[str, str] = {}
    generator_scripts: set[str] = set()
    for definition in commands:
        key = str(definition.key)
        script = str(definition.script)
        generator_scripts.add(script)
        for phrase in str(definition.phrases).split(";"):
            normalised = _normalise_phrase(phrase)
            previous = phrase_owners.get(normalised)
            if previous is not None:
                errors.append(f"Fraza {phrase!r} należy do {previous} i {key}.")
            phrase_owners[normalised] = key

    profile_scripts: set[str] = set()
    profile_phrases: list[str] = []
    confidence_values: set[str] = set()
    for command in profile_commands:
        command_phrases = str(command.findtext("CommandString") or "")
        profile_phrases.extend(
            phrase
            for phrase in command_phrases.split(";")
            if _normalise_phrase(phrase)
        )
        confidence_values.add(str(command.findtext("minimumConfidenceLevel") or ""))
        action = command.find("./ActionSequence/CommandAction")
        if action is None:
            errors.append(f"Komenda {command_phrases!r} nie ma CommandAction.")
            continue
        if action.findtext("ActionType") != "Launch":
            errors.append(f"Komenda {command_phrases!r} nie używa bezpiecznego Launch.")
        context = str(action.findtext("Context") or "")
        script_path = Path(context)
        repo_script_path = SCRIPTS_DIR / script_path.name
        if not script_path.is_file() and not repo_script_path.is_file():
            errors.append(f"Brak skryptu profilu: {context}")
        if script_path.name:
            profile_scripts.add(script_path.name)

    normalised_profile_phrases = [_normalise_phrase(item) for item in profile_phrases]
    if len(normalised_profile_phrases) != len(set(normalised_profile_phrases)):
        errors.append("Profil zawiera zduplikowane frazy.")
    if commands and len(profile_phrases) != len(phrase_owners):
        errors.append(
            f"Generator ma {len(phrase_owners)} fraz, a profil {len(profile_phrases)}."
        )
    if confidence_values and confidence_values != {"65"}:
        errors.append(f"Niespójne progi rozpoznania: {sorted(confidence_values)}")
    if generator_scripts != profile_scripts:
        missing = sorted(generator_scripts - profile_scripts)
        extra = sorted(profile_scripts - generator_scripts)
        errors.append(f"Rozjazd skryptów profilu; brak={missing}, nadmiar={extra}.")

    wrapper_dispatch: dict[str, dict[str, str | None]] = {}
    direct_command_ids: set[str] = set()
    for script in sorted(generator_scripts):
        path = SCRIPTS_DIR / script
        if not path.is_file():
            errors.append(f"Brak wrappera: {path}")
            continue
        dispatch = _read_wrapper(path)
        wrapper_dispatch[script] = dispatch
        command_id = dispatch["command_id"]
        operation = dispatch["operation"]
        if command_id:
            direct_command_ids.add(command_id)
        elif not operation:
            errors.append(f"Wrapper {script} nie ma CommandId ani Operation.")

    capabilities: dict[str, Any] | None = None
    if live:
        try:
            capabilities = _fetch_capabilities(base_url, timeout)
        except RuntimeError as exc:
            errors.append(str(exc))

    runtime_command_ids: set[str] = set()
    runtime_actions: dict[str, dict[str, Any]] = {}
    native_only_actions: list[dict[str, Any]] = []
    if capabilities is not None:
        runtime_command_ids = {
            str(item) for item in capabilities.get("voiceattack_command_ids", [])
        }
        runtime_actions = {
            str(item.get("id")): item
            for item in capabilities.get("voiceattack_actions", [])
            if isinstance(item, dict) and item.get("id")
        }
        native_only_actions = [
            item
            for item in capabilities.get("native_only_actions", [])
            if isinstance(item, dict)
        ]
        if runtime_command_ids != direct_command_ids:
            errors.append(
                "Rozjazd CommandId między wrapperami i żywym katalogiem: "
                f"tylko_wrapper={sorted(direct_command_ids - runtime_command_ids)}, "
                f"tylko_runtime={sorted(runtime_command_ids - direct_command_ids)}."
            )

    for definition in commands:
        script = str(definition.script)
        dispatch = wrapper_dispatch.get(
            script,
            {"command_id": None, "operation": None},
        )
        command_id = dispatch["command_id"]
        action_id = ACTION_ALIASES.get(command_id or "", command_id)
        runtime_action = runtime_actions.get(action_id or "", {})
        inventory.append(
            {
                "key": str(definition.key),
                "spoken_command": str(definition.phrases).split(";")[0],
                "phrase_count": len(str(definition.phrases).split(";")),
                "script": script,
                "dispatch": (
                    f"command:{command_id}"
                    if command_id
                    else f"operation:{dispatch['operation']}"
                ),
                "action_id": action_id,
                "description": str(definition.description),
                "risk": runtime_action.get("risk"),
                "confirmation_required": runtime_action.get(
                    "confirmation_required"
                ),
            }
        )

    voiceattack_running = _voiceattack_running()
    if not voiceattack_running:
        warnings.append(
            "VoiceAttack nie jest uruchomiony; profil na dysku jest poprawny, "
            "ale aktywnego profilu nie da się potwierdzić."
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ok" if not errors else "error",
        "safe_mode": True,
        "profile": {
            "path": str(PROFILE_PATH),
            "name": profile_name,
            "command_count": len(profile_commands),
            "phrase_count": len(profile_phrases),
            "confidence": sorted(confidence_values),
            "missing_scripts": sum(
                1 for item in generator_scripts if not (SCRIPTS_DIR / item).is_file()
            ),
        },
        "runtime": {
            "checked": live,
            "listener_available": capabilities is not None,
            "voiceattack_running": voiceattack_running,
            "active_profile_confirmed": False,
            "direct_command_ids": len(runtime_command_ids or direct_command_ids),
            "voiceattack_actions": len(runtime_actions),
            "native_only_actions": len(native_only_actions),
        },
        "inventory": inventory,
        "native_only_actions": native_only_actions,
        "errors": errors,
        "warnings": warnings,
    }


def _print_report(report: dict[str, Any]) -> None:
    profile = report["profile"]
    runtime = report["runtime"]
    print(
        "AUDYT VOICEATTACK: "
        f"{str(report['status']).upper()} | "
        f"{profile['command_count']} komend | "
        f"{profile['phrase_count']} fraz | "
        f"{profile['missing_scripts']} brakujących skryptów"
    )
    print(
        "RUNTIME: "
        f"listener={'OK' if runtime['listener_available'] else 'BRAK'} | "
        f"VoiceAttack={'URUCHOMIONY' if runtime['voiceattack_running'] else 'WYŁĄCZONY'} | "
        f"akcje VA={runtime['voiceattack_actions']} | "
        f"akcje tylko VoiceLoop={runtime['native_only_actions']}"
    )
    print("\nKOMENDY PROFILU:")
    for index, item in enumerate(report["inventory"], start=1):
        confirmation = (
            " [POTWIERDZENIE]"
            if item.get("confirmation_required")
            else ""
        )
        print(
            f"{index:02d}. {item['spoken_command']} -> "
            f"{item['dispatch']}{confirmation}"
        )
    for warning in report["warnings"]:
        print(f"OSTRZEŻENIE: {warning}")
    for error in report["errors"]:
        print(f"BŁĄD: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audyt profilu VoiceAttack bez wykonywania akcji."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Pomiń porównanie z uruchomionym listenerem VoiceLoop.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Ścieżka raportu JSON.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report = audit_voiceattack_actions(
        live=not args.offline,
        base_url=args.base_url,
        timeout=max(1.0, args.timeout),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_report(report)
    print(f"\nRaport JSON: {args.json}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
