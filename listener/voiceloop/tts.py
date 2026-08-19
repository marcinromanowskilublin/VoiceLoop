from __future__ import annotations

import asyncio
import html
import logging
import os
from typing import Any

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - sprawdzane przez fallback runtime
    speechsdk = None

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
        speaking_rate_percent: int = -20,
        speaking_pitch_percent: int = -5,
    ) -> None:
        self.preferred_voice = preferred_voice
        self.azure_enabled = azure_enabled
        self.azure_key = (azure_key or "").strip()
        self.azure_region = (azure_region or "").strip()
        self.azure_voice = azure_voice.strip() or "pl-PL-ZofiaNeural"
        self.azure_timeout_seconds = max(3.0, azure_timeout_seconds)
        self.speaking_rate_percent = max(-50, min(speaking_rate_percent, 50))
        self.speaking_pitch_percent = max(-50, min(speaking_pitch_percent, 50))
        self._process: asyncio.subprocess.Process | None = None
        self._azure_synthesizer: Any | None = None
        self._azure_speaking = False
        self._stop_requested = False
        self._lock = asyncio.Lock()

    async def speak(self, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        async with self._lock:
            await self.stop()
            self._stop_requested = False
            if self._can_use_azure():
                try:
                    await self._speak_azure(clean_text)
                    return
                except Exception as exc:
                    LOGGER.warning("Azure TTS failed, falling back to Windows voice: %s", exc)
            await self._speak_windows(clean_text)

    async def stop(self) -> None:
        self._stop_requested = True
        synthesizer = self._azure_synthesizer
        if synthesizer is not None and self._azure_speaking:
            await self._stop_azure_synthesizer(synthesizer)
            self._azure_speaking = False

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

    async def _speak_azure(self, text: str) -> None:
        try:
            await self._speak_azure_sdk(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._azure_synthesizer = None
            LOGGER.warning(
                "Azure Speech SDK failed, falling back to Azure REST: %s",
                exc,
            )
            await self._speak_azure_rest(text)

    async def _speak_azure_sdk(self, text: str) -> None:
        synthesizer = self._get_azure_synthesizer()
        ssml = self._azure_ssml(text)
        self._azure_speaking = True
        timeout_seconds = self._speech_timeout_seconds(text, rest_fallback=False)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: synthesizer.speak_ssml_async(ssml).get(),
                ),
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            await self._stop_azure_synthesizer(synthesizer)
            raise
        except TimeoutError as exc:
            await self._stop_azure_synthesizer(synthesizer)
            raise TimeoutError("Azure Speech SDK przekroczył limit czasu.") from exc
        finally:
            self._azure_speaking = False

        if speechsdk is None:
            raise RuntimeError("Pakiet azure-cognitiveservices-speech jest niedostępny.")
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return
        if self._stop_requested:
            return
        if result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            detail = details.error_details or str(details.reason)
            raise RuntimeError(f"Azure Speech SDK anulował syntezę: {detail}")
        raise RuntimeError(f"Azure Speech SDK zwrócił wynik: {result.reason}")

    def _get_azure_synthesizer(self) -> Any:
        if self._azure_synthesizer is not None:
            return self._azure_synthesizer
        if speechsdk is None:
            raise RuntimeError("Brak pakietu azure-cognitiveservices-speech.")

        speech_config = speechsdk.SpeechConfig(
            subscription=self.azure_key,
            region=self.azure_region,
        )
        speech_config.speech_synthesis_voice_name = self.azure_voice
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self._azure_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        return self._azure_synthesizer

    async def _stop_azure_synthesizer(self, synthesizer: Any) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: synthesizer.stop_speaking_async().get(),
                ),
                timeout=3.0,
            )
        except Exception as exc:
            LOGGER.debug("Azure Speech SDK stop failed: %s", exc)

    def _azure_ssml(self, text: str) -> str:
        voice = html.escape(self.azure_voice, quote=True)
        rate = html.escape(self._azure_rate(), quote=True)
        pitch = html.escape(self._azure_pitch(), quote=True)
        content = html.escape(text, quote=False)
        return (
            "<speak version='1.0' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xml:lang='pl-PL'>"
            f"<voice name='{voice}'><prosody rate='{rate}' pitch='{pitch}'>"
            f"{content}</prosody></voice>"
            "</speak>"
        )

    def _azure_rate(self) -> str:
        if self.speaking_rate_percent == 0:
            return "0%"
        return f"{self.speaking_rate_percent:+d}%"

    def _azure_pitch(self) -> str:
        if self.speaking_pitch_percent == 0:
            return "0%"
        return f"{self.speaking_pitch_percent:+d}%"

    def _windows_rate(self) -> int:
        return max(-10, min(round(self.speaking_rate_percent / 10), 10))

    def _speech_timeout_seconds(self, text: str, *, rest_fallback: bool) -> float:
        rate_factor = max(0.5, 1.0 + self.speaking_rate_percent / 100.0)
        characters_per_second = 10.0 * rate_factor
        estimated_playback = len(text) / characters_per_second * 1.25
        margin = 30.0 if rest_fallback else 20.0
        return max(
            self.azure_timeout_seconds,
            min(660.0, estimated_playback + margin),
        )

    async def _speak_windows(self, text: str) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "try{$s.SelectVoice($env:VOICELOOP_TTS_VOICE)}catch{};"
            "$s.Rate=[int]$env:VOICELOOP_TTS_RATE;"
            "$s.Speak($env:VOICELOOP_TTS_TEXT)"
        )
        environment = os.environ.copy()
        environment["VOICELOOP_TTS_TEXT"] = text
        environment["VOICELOOP_TTS_VOICE"] = self.preferred_voice
        environment["VOICELOOP_TTS_RATE"] = str(self._windows_rate())
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

    async def _speak_azure_rest(self, text: str) -> None:
        script = (
            "$ErrorActionPreference='Stop';"
            "$uri='https://' + $env:VOICELOOP_AZURE_REGION + "
            "'.tts.speech.microsoft.com/cognitiveservices/v1';"
            "$headers=@{"
            "'Ocp-Apim-Subscription-Key'=$env:VOICELOOP_AZURE_KEY;"
            "'Content-Type'='application/ssml+xml';"
            "'X-Microsoft-OutputFormat'='riff-24khz-16bit-mono-pcm';"
            "'User-Agent'='VoiceLoop'"
            "};"
            "$ssml=\"<speak version='1.0' xml:lang='pl-PL'>"
            "<voice name='$env:VOICELOOP_AZURE_VOICE'>"
            "<prosody rate='$env:VOICELOOP_AZURE_RATE' pitch='$env:VOICELOOP_AZURE_PITCH'>"
            "$env:VOICELOOP_AZURE_TEXT</prosody></voice></speak>\";"
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
        environment["VOICELOOP_AZURE_TEXT"] = html.escape(text, quote=False)
        environment["VOICELOOP_AZURE_RATE"] = self._azure_rate()
        environment["VOICELOOP_AZURE_PITCH"] = self._azure_pitch()
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
            timeout_seconds = self._speech_timeout_seconds(
                text,
                rest_fallback=True,
            )
            await asyncio.wait_for(self._process.wait(), timeout=timeout_seconds)
            if self._process.returncode not in (0, None):
                raise RuntimeError(f"Azure TTS process exit code: {self._process.returncode}")
        finally:
            await self.stop()
