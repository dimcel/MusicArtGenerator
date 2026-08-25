"""Create deterministic synthetic audio for local synchronization demos.

The generated WAV contains a quiet musical bed and clear beat pulses. It is a
development helper, not part of the installed ``music_art_generator`` package.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import wave
from pathlib import Path
from typing import Iterable


def _soft_clip(value: float) -> float:
    return math.tanh(1.7 * value)


def generate_fake_music_track(
    output_path: str = "test_music.wav",
    beat_frames: Iterable[int] | None = None,
    beat_intensities: Iterable[float] | None = None,
    fps: int = 24,
    total_frames: int = 120,
    sample_rate: int = 44_100,
    profile: str = "base",
) -> str:
    """Generate a mono WAV with deterministic music and beat markers."""
    frames = list(beat_frames or [])
    intensities = list(beat_intensities or [1.0] * len(frames))
    intensities.extend([1.0] * max(0, len(frames) - len(intensities)))

    events = [
        (frame / float(fps), max(0.0, min(1.2, float(intensities[index]))))
        for index, frame in enumerate(frames)
    ]
    total_samples = int((total_frames / float(fps)) * sample_rate)

    random.seed(42)
    pcm: list[int] = []
    for sample in range(total_samples):
        time = sample / float(sample_rate)

        drone = 0.12 * math.sin(2.0 * math.pi * 55.0 * time)
        pad_frequency = 110.0 + 10.0 * math.sin(2.0 * math.pi * 0.07 * time)
        pad = 0.08 * math.sin(2.0 * math.pi * pad_frequency * time)
        hiss = 0.02 * (2.0 * random.random() - 1.0)

        bass_notes = [41.2, 49.0, 55.0, 65.4]
        bass_frequency = bass_notes[int(time * 2.0) % len(bass_notes)]
        bass = 0.10 * math.sin(2.0 * math.pi * bass_frequency * time)

        arp_notes = [220.0, 277.2, 329.6, 440.0]
        arp_frequency = arp_notes[int(time * 8.0) % len(arp_notes)]
        arp_gate = 1.0 if (time * 8.0) % 1.0 < 0.18 else 0.0
        arpeggio = 0.05 * arp_gate * math.sin(2.0 * math.pi * arp_frequency * time)

        beat_signal = 0.0
        for beat_time, intensity in events:
            elapsed = time - beat_time
            if elapsed < 0.0 or elapsed > 0.18:
                continue

            kick_envelope = math.exp(-elapsed * 26.0)
            kick = math.sin(
                2.0 * math.pi * (48.0 - 12.0 * elapsed) * elapsed
            ) * kick_envelope
            snare = (
                (2.0 * random.random() - 1.0)
                * math.exp(-elapsed * 38.0)
                * (0.35 + 0.75 * intensity)
            )
            hat = 0.0
            if elapsed < 0.03:
                hat = (2.0 * random.random() - 1.0) * math.exp(-elapsed * 90.0)
            beat_signal += (0.45 + 0.65 * intensity) * kick + 0.22 * snare + 0.08 * hat

        extra = 0.0
        if profile == "extended":
            offbeat_phase = (time * 4.0) % 1.0
            if offbeat_phase < 0.05:
                extra = (
                    0.06
                    * (2.0 * random.random() - 1.0)
                    * math.exp(-offbeat_phase * 40.0)
                )

        value = _soft_clip(drone + pad + bass + arpeggio + hiss + 0.55 * beat_signal + extra)
        pcm.append(int(max(-1.0, min(1.0, value)) * 32_767.0))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))
    return str(target)


def _regular_beats(total_frames: int, fps: int, bpm: float) -> tuple[list[int], list[float]]:
    frames_per_beat = max(1, round((60.0 / max(1e-6, bpm)) * fps))
    frames = list(range(0, total_frames, frames_per_beat))
    intensities = [0.35 if index % 2 else 1.0 for index in range(len(frames))]
    return frames, intensities


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic audio for local tests.")
    parser.add_argument("--output", default="test_music.wav")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    args = parser.parse_args()

    beat_frames, beat_intensities = _regular_beats(args.frames, args.fps, args.bpm)
    result = generate_fake_music_track(
        output_path=args.output,
        beat_frames=beat_frames,
        beat_intensities=beat_intensities,
        fps=args.fps,
        total_frames=args.frames,
        sample_rate=args.sample_rate,
        profile=args.profile,
    )
    print(f"Created {result}")


if __name__ == "__main__":
    main()
