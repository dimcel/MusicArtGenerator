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

import argparse
import math
import random
import wave
import struct
from typing import Iterable, List, Optional


def _soft_clip(x: float) -> float:
    return math.tanh(1.7 * x)


def _beat_pulse_component(
    phase_seconds: float,
    kick_amp: float,
    snare_amp: float,
) -> float:
    """
    Compact beat pulse made of a decaying kick + noise burst.
    phase_seconds is time since pulse start.
    """
    if phase_seconds < 0.0 or phase_seconds > 0.18:
        return 0.0

    kick_env = math.exp(-phase_seconds * 26.0)
    kick = math.sin(2.0 * math.pi * (56.0 - 11.0 * phase_seconds) * phase_seconds) * kick_env

    snare_env = math.exp(-phase_seconds * 42.0)
    snare = (2.0 * random.random() - 1.0) * snare_env
    return kick_amp * kick + snare_amp * snare


def _write_wav_pcm16(output_path: str, pcm: List[int], sample_rate: int) -> str:
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))
    return output_path


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

    return _write_wav_pcm16(output_path=output_path, pcm=pcm, sample_rate=sample_rate)


def generate_steady_sections_track(
    output_path: str = "fake_steady_sections.wav",
    fps: int = 24,
    total_frames: int = 192,
    sample_rate: int = 44100,
    section_seconds: float = 1.0,
    pulse_every_seconds: float = 0.5,
) -> str:
    """
    Generate deterministic audio with long, steady note sections.

    Useful for testing "stable melody -> directional camera shift" logic.
    Sections rotate through low/high pitch + dark/bright timbres.
    """
    duration = total_frames / float(fps)
    total_samples = int(duration * sample_rate)
    section_seconds = max(0.25, float(section_seconds))
    pulse_every_seconds = max(0.0, float(pulse_every_seconds))

    # (label, base_freq_hz, tone_amp, harmonic_amp, noise_amp)
    # Designed to create distinct pitch/brightness combinations.
    sections = [
        ("left", 115.0, 0.20, 0.06, 0.020),   # low + darker
        ("right", 860.0, 0.12, 0.10, 0.010),  # high + brighter
        ("up", 630.0, 0.15, 0.05, 0.012),     # high-mid + moderate bright
        ("down", 220.0, 0.18, 0.11, 0.018),   # low-mid + brighter harmonic
    ]

    random.seed(123)
    pcm = []
    for n in range(total_samples):
        t = n / float(sample_rate)
        sec_idx = int(t / section_seconds) % len(sections)
        _, f0, tone_amp, harm_amp, noise_amp = sections[sec_idx]
        local_t = t - math.floor(t / section_seconds) * section_seconds

        # Mostly-steady tone with tiny vibrato to stay realistic.
        vib = 1.0 + 0.002 * math.sin(2.0 * math.pi * 0.7 * local_t)
        f = f0 * vib
        tone = tone_amp * math.sin(2.0 * math.pi * f * t)
        harmonic = harm_amp * math.sin(2.0 * math.pi * (2.0 * f) * t)
        sub = 0.40 * tone_amp * math.sin(2.0 * math.pi * (0.5 * f) * t)
        hiss = noise_amp * (2.0 * random.random() - 1.0)

        # Tiny click at section boundaries to mark transitions.
        click = 0.0
        if local_t < 0.022:
            click = 0.22 * math.exp(-local_t * 110.0) * (2.0 * random.random() - 1.0)

        # Optional metronome-like pulse so beat-tracking still sees events.
        pulse = 0.0
        if pulse_every_seconds > 0:
            ph = t % pulse_every_seconds
            if ph < 0.035:
                env = math.exp(-ph * 55.0)
                pulse = 0.16 * env * math.sin(2.0 * math.pi * 92.0 * ph)

        x = tone + harmonic + sub + hiss + click + pulse
        x = _soft_clip(x)
        x = max(-1.0, min(1.0, x))
        pcm.append(int(x * 32767.0))

    return _write_wav_pcm16(output_path=output_path, pcm=pcm, sample_rate=sample_rate)


def _debug_section_label(section_index: int):
    """
    Deterministic section program for debugging:
    - shift_heavy: stable sustained notes (should trigger shift)
    - still: low-energy near-static
    - beat_heavy: strong transient rhythm (should reduce shift)
    - mix: both stable bed + rhythmic hits
    - quiet: low amplitude / near-silent
    """
    program = [
        "shift_heavy",
        "still",
        "beat_heavy",
        "mix",
        "quiet",
        "shift_heavy",
    ]
    return program[section_index % len(program)]


def generate_debug_music_profile_track(
    output_path: str = "fake_debug_profile.wav",
    fps: int = 24,
    total_frames: int = 288,
    sample_rate: int = 44100,
    section_seconds: float = 2.0,
) -> str:
    """
    Generate a sectioned debug track with known behavior every N seconds.

    Default timing (2s sections):
    0-2s  shift_heavy
    2-4s  still
    4-6s  beat_heavy
    6-8s  mix
    8-10s quiet
    10-12s shift_heavy
    """
    duration = total_frames / float(fps)
    total_samples = int(duration * sample_rate)
    section_seconds = max(0.5, float(section_seconds))

    random.seed(321)
    pcm: List[int] = []

    for n in range(total_samples):
        t = n / float(sample_rate)
        sec_idx = int(t / section_seconds)
        sec_label = _debug_section_label(sec_idx)
        sec_local_t = t - sec_idx * section_seconds

        x = 0.0
        if sec_label == "shift_heavy":
            # Stable sustained bed: low onsets, slow variation.
            f = 180.0 + 1.5 * math.sin(2.0 * math.pi * 0.20 * sec_local_t)
            tone = 0.25 * math.sin(2.0 * math.pi * f * t)
            harmonic = 0.08 * math.sin(2.0 * math.pi * (2.0 * f) * t)
            sub = 0.06 * math.sin(2.0 * math.pi * 0.5 * f * t)
            hiss = 0.006 * (2.0 * random.random() - 1.0)
            x = tone + harmonic + sub + hiss

        elif sec_label == "still":
            # Almost static, low-energy region.
            f = 120.0
            tone = 0.035 * math.sin(2.0 * math.pi * f * t)
            hiss = 0.0025 * (2.0 * random.random() - 1.0)
            x = tone + hiss

        elif sec_label == "beat_heavy":
            # Strong transients + moving pitch.
            moving_f = 130.0 + 90.0 * math.sin(2.0 * math.pi * 2.6 * sec_local_t)
            bed = 0.07 * math.sin(2.0 * math.pi * moving_f * t)
            beat_phase = sec_local_t % 0.5
            beat = _beat_pulse_component(beat_phase, kick_amp=0.52, snare_amp=0.24)
            hiss = 0.013 * (2.0 * random.random() - 1.0)
            x = bed + beat + hiss

        elif sec_label == "mix":
            # Stable pad + beat pulses together.
            f = 220.0 + 2.0 * math.sin(2.0 * math.pi * 0.35 * sec_local_t)
            pad = 0.16 * math.sin(2.0 * math.pi * f * t)
            harmonic = 0.05 * math.sin(2.0 * math.pi * (2.0 * f) * t)
            beat_phase = sec_local_t % 0.5
            beat = _beat_pulse_component(beat_phase, kick_amp=0.26, snare_amp=0.11)
            hiss = 0.008 * (2.0 * random.random() - 1.0)
            x = pad + harmonic + beat + hiss

        else:  # quiet
            # Quiet part with explicit soft in/out inside the section.
            # This makes quiet-hold fade behavior easier to validate visually.
            fade_seconds = min(0.45, 0.22 * section_seconds)
            if fade_seconds <= 1e-6:
                shape = 1.0
            else:
                in_gain = min(1.0, sec_local_t / fade_seconds)
                out_gain = min(1.0, (section_seconds - sec_local_t) / fade_seconds)
                shape = max(0.0, min(in_gain, out_gain))

            env = 0.008 + 0.022 * shape
            f = 160.0 + 2.0 * math.sin(2.0 * math.pi * 0.22 * sec_local_t)
            tone = env * math.sin(2.0 * math.pi * f * t)
            hiss = (0.0018 + 0.0012 * shape) * (2.0 * random.random() - 1.0)
            x = tone + hiss

        x = _soft_clip(x)
        x = max(-1.0, min(1.0, x))
        pcm.append(int(x * 32767.0))

    return _write_wav_pcm16(output_path=output_path, pcm=pcm, sample_rate=sample_rate)


def _regular_beats(total_frames: int, fps: int, bpm: float):
    frames_per_beat = max(1.0, (60.0 / max(1e-6, bpm)) * fps)
    beats = list(range(0, total_frames, int(frames_per_beat)))
    intensities = [0.35 if i % 2 else 1.0 for i in range(len(beats))]
    return beats, intensities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["music", "steady", "debug"], default="steady")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames", type=int, default=192)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    parser.add_argument(
        "--section-seconds",
        type=float,
        default=1.0,
        help="Duration of each steady-note block in steady mode.",
    )
    parser.add_argument(
        "--pulse-every-seconds",
        type=float,
        default=0.5,
        help="Metronome pulse interval in steady mode. 0 disables pulses.",
    )
    parser.add_argument(
        "--debug-section-seconds",
        type=float,
        default=2.0,
        help="Section length for debug mode timeline.",
    )
    args = parser.parse_args()

    if args.mode == "music":
        beats, intensities = _regular_beats(args.frames, fps=args.fps, bpm=args.bpm)
        out = generate_fake_music_track(
            output_path=args.output or "fake_music.wav",
            beat_frames=beats,
            beat_intensities=intensities,
            fps=args.fps,
            total_frames=args.frames,
            sample_rate=args.sample_rate,
            profile=args.profile,
        )
    else:
        if args.mode == "steady":
            out = generate_steady_sections_track(
                output_path=args.output or "fake_steady_sections.wav",
                fps=args.fps,
                total_frames=args.frames,
                sample_rate=args.sample_rate,
                section_seconds=args.section_seconds,
                pulse_every_seconds=args.pulse_every_seconds,
            )
        else:
            out = generate_debug_music_profile_track(
                output_path=args.output or "fake_debug_profile.wav",
                fps=args.fps,
                total_frames=args.frames,
                sample_rate=args.sample_rate,
                section_seconds=args.debug_section_seconds,
            )
            duration = args.frames / float(args.fps)
            num_sections = int(math.ceil(duration / float(max(0.5, args.debug_section_seconds))))
            print("Debug timeline:")
            for i in range(num_sections):
                start = i * float(args.debug_section_seconds)
                end = min(duration, (i + 1) * float(args.debug_section_seconds))
                start_frame = int(round(start * args.fps))
                end_frame = min(args.frames - 1, int(round(end * args.fps)) - 1)
                print(
                    f"  {start:>4.1f}s - {end:>4.1f}s "
                    f"(f{start_frame:03d}-f{end_frame:03d}) : {_debug_section_label(i)}"
                )

    print(f"Created {out}")
