from __future__ import annotations

import asyncio
import html
import logging
import os

LOGGER = logging.getLogger("voiceloop.tts")


class WindowsTTS:
    def __init__(
        self,
        preferred_voice: str = "Microsoft Paulina Desktop",
        *,
        azure_enabled: bool = False,
        azure_key: str | None = None,
        azure_region: str | None = None,
        azure_voice: str = "pl-PL-ZofiaNeural",
        azure_timeout_seconds: float = 20.0,
    ) -> None:
        self.preferred_voice = preferred_voice
        self.azure_enabled = azure_enabled
        self.azure_key = (azure_key or "").strip()
        self.azure_region = (azure_region or "").strip()
        self.azure_voice = azure_voice.strip() or "pl-PL-ZofiaNeural"
        self.azure_timeout_seconds = max(3.0, azure_timeout_seconds)
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def speak(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        async with self._lock:
            await self.stop()
            if self._can_use_azure():
                try:
                    await self._speak_azure(clean_text)
                    return
                except Exception as exc:
                    LOGGER.warning("Azure TTS failed, falling back to Windows voice: %s", exc)
            await self._speak_windows(clean_text)

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

    def _can_use_azure(self) -> bool:
        return bool(self.azure_enabled and self.azure_key and self.azure_region)

    async def _speak_windows(self, text: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "try{$s.SelectVoice($env:VOICELOOP_TTS_VOICE)}catch{};"
            "$s.Speak($env:VOICELOOP_TTS_TEXT)"
        )
        environment = os.environ.copy()
        environment["VOICELOOP_TTS_TEXT"] = text[:4000]
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

    async def _speak_azure(self, text: str) -> None:
        script = (
            "$ErrorActionPreference='Stop';"
            "$uri='https://' + $env:VOICELOOP_AZURE_REGION + '.tts.speech.microsoft.com/cognitiveservices/v1';"
            "$headers=@{"
            "'Ocp-Apim-Subscription-Key'=$env:VOICELOOP_AZURE_KEY;"
            "'Content-Type'='application/ssml+xml';"
            "'X-Microsoft-OutputFormat'='riff-24khz-16bit-mono-pcm';"
            "'User-Agent'='VoiceLoop'"
            "};"
            "$ssml=\"<speak version='1.0' xml:lang='pl-PL'><voice name='$env:VOICELOOP_AZURE_VOICE'>$env:VOICELOOP_AZURE_TEXT</voice></speak>\";"
            "$tmp=[System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(),'wav');"
            "try{"
            "Invoke-WebRequest -Method Post -Uri $uri -Headers $headers -Body $ssml -OutFile $tmp;"
            "$player=New-Object System.Media.SoundPlayer $tmp;"
            "$player.PlaySync()"
            "}finally{"
            "if(Test-Path $tmp){Remove-Item $tmp -Force -ErrorAction SilentlyContinue}"
            "}"
        )
        environment = os.environ.copy()
        environment["VOICELOOP_AZURE_KEY"] = self.azure_key
        environment["VOICELOOP_AZURE_REGION"] = self.azure_region
        environment["VOICELOOP_AZURE_VOICE"] = self.azure_voice
        environment["VOICELOOP_AZURE_TEXT"] = html.escape(text[:3000], quote=False)
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
            await asyncio.wait_for(self._process.wait(), timeout=self.azure_timeout_seconds)
            if self._process.returncode not in (0, None):
                raise RuntimeError(f"Azure TTS process exit code: {self._process.returncode}")
        finally:
            await self.stop()
