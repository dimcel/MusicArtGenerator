"""
Fake Music / Noise Generator for Sync Testing.

    Creates a synthetic WAV track with:
- low drone
- noise texture
- beat-aligned kick/snare/hat pulses
- moving bassline
- simple arpeggio sparkle
- occasional sweep risers

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
    profile: str = "base",
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

        # Musical layers for clearer sync perception.
        beat_index = int(t * 2.0)  # slow harmonic movement
        bass_notes = [41.2, 49.0, 55.0, 65.4]  # E1, G1, A1, C2
        bass_f = bass_notes[beat_index % len(bass_notes)]
        bass = 0.10 * math.sin(2.0 * math.pi * bass_f * t)

        arp_notes = [220.0, 277.2, 329.6, 440.0]
        arp_step = int(t * 8.0)
        arp_f = arp_notes[arp_step % len(arp_notes)]
        arp_gate = 1.0 if (t * 8.0) % 1.0 < 0.18 else 0.0
        arp = 0.05 * arp_gate * math.sin(2.0 * math.pi * arp_f * t)

        # Extra rhythmic content for extended profile.
        extra = 0.0
        if profile == "extended":
            # Off-beat clap/noise hit every half beat.
            offbeat_phase = (t * 4.0) % 1.0
            if offbeat_phase < 0.05:
                extra += 0.06 * (2.0 * random.random() - 1.0) * math.exp(-offbeat_phase * 40.0)

            # Slow wobble layer to make section changes obvious.
            wobble = 0.04 * math.sin(2.0 * math.pi * (0.18 + 0.06 * math.sin(2 * math.pi * 0.03 * t)) * t)
            extra += wobble

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

            # Small riser before stronger beats.
            if inten >= 0.8:
                pre = bt - t
                if 0.0 < pre < 0.16:
                    sweep_f = 350.0 + 850.0 * (1.0 - pre / 0.16)
                    sweep_env = (1.0 - pre / 0.16) ** 2
                    beat_sig += 0.05 * sweep_env * math.sin(2.0 * math.pi * sweep_f * t)

        x = drone + pad + bass + arp + hiss + 0.55 * beat_sig + extra
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
