from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlparse

from .schema import StyleEvaluationReport, StyleProfile


class LocalOnlyViolation(RuntimeError):
    pass


def require_loopback_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().casefold()
    if parsed.scheme not in {"http", "https"} or not host:
        raise LocalOnlyViolation("Lokalny endpoint musi być poprawnym URL HTTP(S).")
    if host == "localhost":
        return url.rstrip("/")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise LocalOnlyViolation("Endpoint korpusu musi wskazywać loopback.") from exc
    if not address.is_loopback:
        raise LocalOnlyViolation("Endpoint korpusu musi wskazywać loopback.")
    return url.rstrip("/")


def load_style_profile(path: Path, *, enabled: bool) -> StyleProfile | None:
    if not enabled or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = StyleProfile.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise LocalOnlyViolation("Nieprawidłowy lokalny profil stylu.") from exc
    if not profile.enabled:
        return None
    report_path = path.parent / "holdout-report-v1.json"
    try:
        report = StyleEvaluationReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise LocalOnlyViolation("Brak poprawnego raportu holdout profilu stylu.") from exc
    if report.profile_id != profile.profile_id or not report.passes_quality_gate:
        raise LocalOnlyViolation("Profil stylu nie przeszedł lokalnej bramki jakości.")
    return profile
