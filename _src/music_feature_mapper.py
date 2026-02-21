"""
Music Feature Mapper (Simple)

Maps frame-aligned audio features to video generation controls.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MusicMappingConfig:
    # Diffusion params
    strength_base: float = 0.58
    strength_energy_boost: float = 0.22
    strength_beat_boost: float = 0.12

    cfg_base: float = 7.0
    cfg_brightness_boost: float = 2.0
    cfg_beat_boost: float = 1.6

    # Camera deltas applied before img2img
    zoom_base: float = 1.003
    zoom_energy_boost: float = 0.010
    zoom_beat_boost: float = 0.012

    pan_base: float = 0.0
    pan_onset_boost: float = 3.0
    pan_wave_amp: float = 2.5

    angle_base: float = -0.05
    angle_onset_boost: float = 3.0
    angle_wave_amp: float = 1.8

    # Clamp ranges
    strength_min: float = 0.42
    strength_max: float = 0.97
    cfg_min: float = 5.8
    cfg_max: float = 13.5
    zoom_min: float = 0.995
    zoom_max: float = 1.05
    pan_abs_max: float = 10.0
    angle_abs_max: float = 8.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def map_frame_to_controls(
    frame: int,
    total_frames: int,
    energy: float,
    onset: float,
    brightness: float,
    beat_pulse: float,
    cfg: MusicMappingConfig = None,
) -> Dict[str, float]:
    """
    Return controls for one frame.
    """
    import math

    if cfg is None:
        cfg = MusicMappingConfig()

    t = 0.0 if total_frames <= 1 else frame / float(total_frames - 1)

    strength = (
        cfg.strength_base
        + cfg.strength_energy_boost * energy
        + cfg.strength_beat_boost * beat_pulse
    )
    strength = _clamp(strength, cfg.strength_min, cfg.strength_max)

    cfg_scale = (
        cfg.cfg_base
        + cfg.cfg_brightness_boost * brightness
        + cfg.cfg_beat_boost * beat_pulse
    )
    cfg_scale = _clamp(cfg_scale, cfg.cfg_min, cfg.cfg_max)

    zoom_delta = (
        cfg.zoom_base
        + cfg.zoom_energy_boost * energy
        + cfg.zoom_beat_boost * beat_pulse
    )
    zoom_delta = _clamp(zoom_delta, cfg.zoom_min, cfg.zoom_max)

    tx_delta = (
        cfg.pan_base
        + cfg.pan_wave_amp * math.sin(2.0 * math.pi * t * 2.0)
        + cfg.pan_onset_boost * (onset - 0.25)
    )
    tx_delta = _clamp(tx_delta, -cfg.pan_abs_max, cfg.pan_abs_max)

    angle_delta = (
        cfg.angle_base
        + cfg.angle_wave_amp * math.sin(2.0 * math.pi * t * 1.4)
        + cfg.angle_onset_boost * beat_pulse
    )
    angle_delta = _clamp(angle_delta, -cfg.angle_abs_max, cfg.angle_abs_max)

    # Prompt profile index (0..3): small on calm frames, bigger on intense frames.
    prompt_drive = 0.45 * energy + 0.35 * brightness + 0.20 * beat_pulse
    prompt_level = int(_clamp(prompt_drive * 4.0, 0, 3))

    # Seed drift: more jumps when onsets/beat spike.
    seed_jump = int(100 * onset + 120 * beat_pulse)

    return {
        "strength": strength,
        "cfg_scale": cfg_scale,
        "zoom_delta": zoom_delta,
        "tx_delta": tx_delta,
        "ty_delta": 0.35 * tx_delta,
        "angle_delta": angle_delta,
        "prompt_level": prompt_level,
        "seed_jump": seed_jump,
    }
