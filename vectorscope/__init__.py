"""Vectorscope — laboratorium embeddingów wpięte w środowisko VoiceLoopa.

Panel celowo nie ma własnego venva ani własnego `.env`. Importuje moduły
VoiceLoopa wprost z `listener/`, żeby mierzyć dokładnie ten sam klient
embeddingów, te same progi i tę samą pamięć, których używa asystent. Kopia
konfiguracji rozjechałaby się po pierwszej zmianie i panel zacząłby opisywać
system, który nie istnieje.
"""

from __future__ import annotations

import sys
from pathlib import Path

VECTORSCOPE_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LISTENER_DIR = PROJECT_ROOT / "listener"

if str(LISTENER_DIR) not in sys.path:
    sys.path.insert(0, str(LISTENER_DIR))

__all__ = ["LISTENER_DIR", "PROJECT_ROOT", "VECTORSCOPE_VERSION"]
