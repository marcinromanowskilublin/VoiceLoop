"""Run VoiceLoop's public, deterministic context-evidence scenario."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTENER = ROOT / "listener"
if str(LISTENER) not in sys.path:
    sys.path.insert(0, str(LISTENER))

from voiceloop.context.evidence_demo import build_context_evidence_demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-network SQLite + Qdrant-local evidence scenario and "
            "write JSON, Markdown and HTML reports."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "context-evidence-demo",
    )
    parser.add_argument("--revision", default=None)
    return parser


def _revision(value: str | None) -> str:
    if value:
        return value
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = asyncio.run(
        build_context_evidence_demo(
            output_dir=args.output,
            revision=_revision(args.revision),
        )
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "source_revision": report["source_revision"],
                "qdrant_points": report["storage"]["qdrant_points"],
                "verdicts": {
                    row["name"]: row["verdict"] for row in report["scenarios"]
                },
                "promotion_allowed": report["evaluation"]["promotion_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
