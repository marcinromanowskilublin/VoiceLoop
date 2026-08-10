"""Compatibility notice for the retired standalone Deepgram listener."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Ten listener jest wycofany. Uruchom rdzeń przez scripts\\start-core.bat "
        "i steruj nasłuchem w panelu http://127.0.0.1:8765."
    )


if __name__ == "__main__":
    main()
