from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .schema import ProsodyFeaturesV1, VoiceEvalSampleV1


class ProsodyError(RuntimeError):
    pass


def analyze_prosody(
    sample: VoiceEvalSampleV1,
    *,
    eval_root: Path,
    word_count: int | None = None,
) -> ProsodyFeaturesV1:
    if sample.audio is None:
        return _unavailable(sample.sample_id, "missing_audio")
    audio_path = _safe_audio_path(eval_root, sample.audio.relative_path)
    try:
        signal, sample_rate = _read_wav(audio_path)
    except (OSError, wave.Error, ProsodyError) as exc:
        return _unavailable(sample.sample_id, f"invalid_audio:{type(exc).__name__}")
    duration = len(signal) / sample_rate if sample_rate else 0.0
    if duration < 0.35:
        return _unavailable(sample.sample_id, "audio_too_short", duration=duration)

    frame_length = max(1, round(sample_rate * 0.04))
    hop_length = max(1, round(sample_rate * 0.01))
    frames = _frames(signal, frame_length, hop_length)
    if frames.size == 0:
        return _unavailable(sample.sample_id, "no_frames", duration=duration)

    window = np.hanning(frame_length)
    windowed = frames * window
    rms = np.sqrt(np.mean(np.square(windowed), axis=1))
    peak_rms = float(np.max(rms)) if len(rms) else 0.0
    noise_floor = float(np.quantile(rms, 0.20)) if len(rms) else 0.0
    speech_threshold = max(
        min(noise_floor * 2.5, peak_rms * 0.30),
        peak_rms * 0.05,
        1e-5,
    )
    speech_mask = rms >= speech_threshold
    silence_ratio = float(1.0 - np.mean(speech_mask))
    pause_count, pause_seconds = _pause_metrics(
        speech_mask,
        hop_seconds=hop_length / sample_rate,
    )

    f0_values: list[float] = []
    f0_times: list[float] = []
    for index, frame in enumerate(windowed):
        if not speech_mask[index]:
            continue
        f0, periodicity = _estimate_f0(frame, sample_rate)
        if f0 is not None and periodicity >= 0.30:
            f0_values.append(f0)
            f0_times.append((index * hop_length + frame_length / 2) / sample_rate)

    voiced_ratio = len(f0_values) / max(1, int(np.sum(speech_mask)))
    if len(f0_values) < 5 or voiced_ratio < 0.12:
        return ProsodyFeaturesV1(
            sample_id=sample.sample_id,
            available=False,
            reason="insufficient_voiced_audio",
            duration_seconds=duration,
            voiced_ratio=max(0.0, min(voiced_ratio, 1.0)),
            rms_mean=float(np.mean(rms)),
            rms_peak=peak_rms,
            final_rms_delta=_final_rms_delta(rms),
            pause_count=pause_count,
            pause_total_seconds=pause_seconds,
            silence_ratio=max(0.0, min(silence_ratio, 1.0)),
            words_per_second=(word_count / duration if word_count is not None else None),
            confidence=0.0,
        )

    f0_array = np.asarray(f0_values, dtype=np.float64)
    time_array = np.asarray(f0_times, dtype=np.float64)
    median_f0 = float(np.median(f0_array))
    semitones = 12.0 * np.log2(f0_array / median_f0)
    f0_range = float(np.quantile(semitones, 0.9) - np.quantile(semitones, 0.1))
    final_mask = time_array >= max(0.0, duration - 0.8)
    final_times = time_array[final_mask]
    final_semitones = semitones[final_mask]
    final_delta: float | None = None
    final_slope: float | None = None
    if len(final_semitones) >= 3:
        split = max(1, len(final_semitones) // 3)
        final_delta = float(
            np.median(final_semitones[-split:]) - np.median(final_semitones[:split])
        )
        centered_time = final_times - np.mean(final_times)
        denominator = float(np.dot(centered_time, centered_time))
        if denominator > 0:
            final_slope = float(
                np.dot(centered_time, final_semitones - np.mean(final_semitones))
                / denominator
            )

    confidence = min(
        1.0,
        0.35 + 0.45 * min(voiced_ratio / 0.55, 1.0) + 0.20 * min(duration / 2.0, 1.0),
    )
    return ProsodyFeaturesV1(
        sample_id=sample.sample_id,
        available=True,
        duration_seconds=duration,
        voiced_ratio=max(0.0, min(voiced_ratio, 1.0)),
        f0_median_hz=median_f0,
        f0_range_semitones=max(0.0, f0_range),
        final_f0_delta_semitones=final_delta,
        final_f0_slope_semitones_per_second=final_slope,
        rms_mean=float(np.mean(rms)),
        rms_peak=peak_rms,
        final_rms_delta=_final_rms_delta(rms),
        pause_count=pause_count,
        pause_total_seconds=pause_seconds,
        silence_ratio=max(0.0, min(silence_ratio, 1.0)),
        words_per_second=(word_count / duration if word_count is not None else None),
        confidence=confidence,
    )


def question_intonation_score(features: ProsodyFeaturesV1) -> float:
    if not features.available:
        return 0.0
    delta = features.final_f0_delta_semitones or 0.0
    slope = features.final_f0_slope_semitones_per_second or 0.0
    rise_score = _sigmoid((delta - 0.8) * 1.2)
    slope_score = _sigmoid((slope - 1.0) * 0.35)
    confidence = features.confidence
    return max(0.0, min(1.0, (rise_score * 0.6 + slope_score * 0.4) * confidence))


def _safe_audio_path(eval_root: Path, relative_path: str) -> Path:
    root = eval_root.resolve()
    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProsodyError("Audio jest poza katalogiem ewaluacji.") from exc
    return candidate


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ProsodyError("Wymagane jest audio PCM 16-bit.")
    values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return values, sample_rate


def _frames(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(signal) < frame_length:
        padded = np.pad(signal, (0, frame_length - len(signal)))
        return padded.reshape(1, -1)
    frame_count = 1 + (len(signal) - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def _estimate_f0(frame: np.ndarray, sample_rate: int) -> tuple[float | None, float]:
    centered = frame - np.mean(frame)
    energy = float(np.dot(centered, centered))
    if energy <= 1e-8:
        return None, 0.0
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    min_lag = max(1, sample_rate // 400)
    max_lag = min(len(correlation) - 1, sample_rate // 70)
    if max_lag <= min_lag:
        return None, 0.0
    search = correlation[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(search))
    periodicity = float(correlation[lag] / max(correlation[0], 1e-12))
    if periodicity <= 0:
        return None, periodicity
    return sample_rate / lag, periodicity


def _pause_metrics(mask: np.ndarray, *, hop_seconds: float) -> tuple[int, float]:
    count = 0
    total = 0.0
    run = 0
    minimum_frames = max(1, round(0.20 / hop_seconds))
    for active in mask:
        if active:
            if run >= minimum_frames:
                count += 1
                total += run * hop_seconds
            run = 0
        else:
            run += 1
    if run >= minimum_frames:
        count += 1
        total += run * hop_seconds
    return count, total


def _final_rms_delta(rms: np.ndarray) -> float | None:
    if len(rms) < 6:
        return None
    width = max(1, len(rms) // 6)
    return float(np.mean(rms[-width:]) - np.mean(rms[-2 * width : -width]))


def _unavailable(
    sample_id: str,
    reason: str,
    *,
    duration: float = 0.0,
) -> ProsodyFeaturesV1:
    return ProsodyFeaturesV1(
        sample_id=sample_id,
        available=False,
        reason=reason,
        duration_seconds=max(0.0, duration),
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
