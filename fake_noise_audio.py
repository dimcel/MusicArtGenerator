"""
Fake Music / Noise Generator for Sync Testing.

Creates a synthetic WAV track with:
- low drone
- noise texture
- beat-aligned kick/snare/hat pulses

Designed for quick A/V sync tests without real music files.
"""

import math
import random
import wave
import struct
from typing import Iterable, List, Optional


def _soft_clip(x: float) -> float:
    return math.tanh(1.7 * x)


def generate_fake_music_track(
    output_path: str = "fake_music.wav",
    beat_frames: Optional[Iterable[int]] = None,
    beat_intensities: Optional[Iterable[float]] = None,
    fps: int = 24,
    total_frames: int = 120,
    sample_rate: int = 44100,
) -> str:
    """
    Generate a synthetic beat-driven audio file.

    Args:
        output_path: Target wav file.
        beat_frames: Iterable of beat frame indices.
        beat_intensities: Same length as beat_frames, values ~0..1.
        fps: Video FPS for frame->time conversion.
        total_frames: Total video frames.
        sample_rate: Audio sample rate.
    """
    beat_frames = list(beat_frames or [])
    beat_intensities = list(beat_intensities or [1.0] * len(beat_frames))
    if len(beat_intensities) < len(beat_frames):
        pad = [1.0] * (len(beat_frames) - len(beat_intensities))
        beat_intensities.extend(pad)

    duration = total_frames / float(fps)
    total_samples = int(duration * sample_rate)

    beat_events = []
    for i, frame in enumerate(beat_frames):
        t = frame / float(fps)
        intensity = max(0.0, min(1.2, float(beat_intensities[i])))
        beat_events.append((t, intensity))

    random.seed(42)
    pcm = []

    for n in range(total_samples):
        t = n / float(sample_rate)

        # Ambient bed.
        drone = 0.12 * math.sin(2.0 * math.pi * 55.0 * t)
        pad = 0.08 * math.sin(2.0 * math.pi * (110.0 + 10.0 * math.sin(2 * math.pi * 0.07 * t)) * t)
        hiss = 0.02 * (2.0 * random.random() - 1.0)

        beat_sig = 0.0
        for bt, inten in beat_events:
            dt = t - bt
            if dt < 0 or dt > 0.18:
                continue

            # Kick (short decaying low thump).
            kick_env = math.exp(-dt * 26.0)
            kick = math.sin(2.0 * math.pi * (48.0 - 12.0 * dt) * dt) * kick_env

            # Snare-ish noise burst (big beats stronger).
            snare_env = math.exp(-dt * 38.0)
            snare = (2.0 * random.random() - 1.0) * snare_env * (0.35 + 0.75 * inten)

            # Hat click at very beginning.
            hat = 0.0
            if dt < 0.03:
                hat = (2.0 * random.random() - 1.0) * math.exp(-dt * 90.0)

            beat_sig += (0.45 + 0.65 * inten) * kick + 0.22 * snare + 0.08 * hat

        x = drone + pad + hiss + 0.55 * beat_sig
        x = _soft_clip(x)
        x = max(-1.0, min(1.0, x))
        pcm.append(int(x * 32767.0))

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))

    return output_path


if __name__ == "__main__":
    # Standalone default test.
    # 5 seconds at 24fps with regular beats every ~12 frames.
    beats = list(range(0, 120, 12))
    intensities: List[float] = [0.3 if i % 2 else 1.0 for i in range(len(beats))]
    out = generate_fake_music_track(
        output_path="fake_music.wav",
        beat_frames=beats,
        beat_intensities=intensities,
        fps=24,
        total_frames=120,
    )
    print(f"Created {out}")

