# AGENTS.md

## Cursor Cloud specific instructions

VoiceLoop is a **Windows-first** Python 3.11 project (FastAPI core in `listener/`).
CI runs on `windows-latest`. On the Linux Cloud VM the core still runs, but in
**degraded mode**: the Windows-only OS automation (`pywin32`/`pywinauto`) and
external providers (LM Studio, Qdrant, Deepgram, Screenpipe) are unavailable, so
`/api/v1/health` reports `degraded`. That is expected here and is not a bug — the
core, memory, routing, corpus, and HTTP API all work. See `README.md` for the
canonical command list; only the non-obvious Linux caveats are documented below.

### Environment already provided by the snapshot
- Python 3.11 (via deadsnakes), plus PortAudio, PulseAudio, `libsndfile`,
  `libasound2`. The update script only refreshes the `listener/.venv` on top of
  these; it does not install system packages.
- The Linux venv intentionally **excludes** `pywin32`, `pywinauto`, and
  `comtypes` (Windows-only). Their code paths are lazily imported and only run
  for real Windows OS actions, so importing/running on Linux is fine.

### Critical gotcha: PulseAudio must be running before import
`voiceloop/audio_capture.py` does `import soundcard`, which **connects to a
PulseAudio server at import time**. Since `app.py` and `meeting_recorder.py`
import it, both pytest collection and the uvicorn server will crash on Linux
unless a Pulse server is running and `XDG_RUNTIME_DIR` is set. Start it once per
boot before running tests or the server:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
pulseaudio --start --exit-idle-time=-1
```

(This is a per-boot runtime process; it is not preserved as a running process by
the snapshot.)

### Lint / test (run from `listener/`, with the venv + PulseAudio)
```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
./.venv/bin/python -m ruff check voiceloop ../tests \
  ../scripts/voice_capture_server.py \
  ../scripts/holding-commands/server.py \
  ../scripts/calibration-phrases/server.py
./.venv/bin/python -m pytest -c pyproject.toml -q
```
Baseline: **541 passed, 1 skipped** (the skipped test is the intentional
private-data replay).

### Run the core app
```bash
cd listener
export XDG_RUNTIME_DIR=/run/user/$(id -u)
./.venv/bin/python -m uvicorn voiceloop.app:app --host 127.0.0.1 --port 8765
```
- Panel: `http://127.0.0.1:8765/`
- Private endpoints need the header `X-VoiceLoop-Token`. Get the token from the
  loopback-only `GET /api/v1/session`, e.g. create a memory:
  ```bash
  TOKEN=$(curl -s http://127.0.0.1:8765/api/v1/session | sed -E 's/.*"token":"([^"]+)".*/\1/')
  curl -s -X POST http://127.0.0.1:8765/api/v1/memories \
    -H "X-VoiceLoop-Token: $TOKEN" -H "Content-Type: application/json" \
    -d '{"content":"hello","kind":"note"}'
  ```
- `listener/.env` is optional (gitignored); defaults are local-first. Copy from
  `listener/.env.example` if you need to set provider keys.
