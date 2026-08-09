from __future__ import annotations

import asyncio
import os


class WindowsTTS:
    def __init__(self, preferred_voice: str = "Microsoft Paulina Desktop") -> None:
        self.preferred_voice = preferred_voice
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def speak(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        async with self._lock:
            await self.stop()
            script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "try{$s.SelectVoice($env:VOICELOOP_TTS_VOICE)}catch{};"
                "$s.Speak($env:VOICELOOP_TTS_TEXT)"
            )
            environment = os.environ.copy()
            environment["VOICELOOP_TTS_TEXT"] = clean_text[:4000]
            environment["VOICELOOP_TTS_VOICE"] = self.preferred_voice
            self._process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
                env=environment,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await self._process.wait()
            finally:
                self._process = None

    async def stop(self) -> None:
        process = self._process
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        self._process = None
