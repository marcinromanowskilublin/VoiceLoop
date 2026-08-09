# VoiceLoop: lokalny listener Deepgram
# mikrofon -> WebSocket (Deepgram live) -> finalny transkrypt -> HTTP POST -> webhook n8n
#
# Uruchamianie: start-listener.bat (tworzy venv i instaluje zaleznosci przy 1. uruchomieniu)

import asyncio
import json
import os
import sys
import urllib.request

import sounddevice as sd
import websockets

# --- konfiguracja ---------------------------------------------------------
MODEL = "nova-3"          # przy braku PL w nova-3 skrypt sam sprobuje nova-2
LANGUAGE = "pl"
SAMPLE_RATE = 16000
WEBHOOK = "http://localhost:5678/webhook/voice"
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPGRAM_API_KEY="):
                    return line.split("=", 1)[1].strip()
    print("[blad] Brak klucza. Wpisz DEEPGRAM_API_KEY=... w pliku listener\\.env")
    sys.exit(1)


def build_url(model: str) -> str:
    params = (
        f"model={model}&language={LANGUAGE}"
        f"&encoding=linear16&sample_rate={SAMPLE_RATE}&channels=1"
        "&smart_format=true&punctuate=true"
        "&interim_results=true&endpointing=300&utterance_end_ms=1200"
    )
    return f"wss://api.deepgram.com/v1/listen?{params}"


def post_to_n8n(text: str) -> None:
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        print(f"[-> n8n] {text}")
    except Exception as e:
        print(f"[blad] n8n nie odpowiada ({e}). Czy dziala scripts\\start-n8n.bat?")


async def run(key: str, model: str) -> None:
    url = build_url(model)
    # autoryzacja przez subprotokol 'token' (dziala w kazdej wersji websockets)
    async with websockets.connect(url, subprotocols=["token", key]) as ws:
        print(f"[ok] Polaczono z Deepgram ({model}, {LANGUAGE}). Mow do mikrofonu...")
        print("[i] Po pauzie w mowieniu zdanie leci do n8n. Ctrl+C konczy.\n")

        loop = asyncio.get_running_loop()
        audio_q: asyncio.Queue = asyncio.Queue()
        finals: list[str] = []

        def mic_callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

        def flush():
            text = " ".join(finals).strip()
            finals.clear()
            if text:
                post_to_n8n(text)

        async def sender():
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=SAMPLE_RATE // 10,  # 100 ms
                callback=mic_callback,
            )
            with stream:
                while True:
                    chunk = await audio_q.get()
                    await ws.send(chunk)

        async def receiver():
            async for raw in ws:
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "Results":
                    alts = (d.get("channel") or {}).get("alternatives") or [{}]
                    t = (alts[0].get("transcript") or "").strip()
                    if d.get("is_final"):
                        if t:
                            finals.append(t)
                            print(f"[final] {t}")
                        if d.get("speech_final"):
                            flush()
                    elif t:
                        print(f"[...] {t}", end="\r")
                elif d.get("type") == "UtteranceEnd":
                    flush()

        await asyncio.gather(sender(), receiver())


def main() -> None:
    key = load_api_key()
    try:
        asyncio.run(run(key, MODEL))
    except KeyboardInterrupt:
        print("\n[koniec] Zatrzymano.")
    except websockets.exceptions.InvalidStatus as e:
        print(f"[uwaga] Deepgram odrzucil polaczenie ({e}). Probuje modelu nova-2...")
        try:
            asyncio.run(run(key, "nova-2"))
        except KeyboardInterrupt:
            print("\n[koniec] Zatrzymano.")


if __name__ == "__main__":
    main()
