"""
Prompt blending helpers.

Keeps prompt logic out of test scripts and supports:
- discrete prompt selection
- blended prompt transition between nearby concepts
"""

from dataclasses import dataclass
import math


@dataclass
class PromptBlendConfig:
    subjects: tuple = (
        "middle-aged man sitting on a wooden bench",
        "middle-aged man resting on a park bench",
        "middle-aged man seated quietly on a bench",
        "middle-aged man on a bench, thoughtful expression",
    )
    scenes: tuple = (
        "in an autumn park with soft afternoon light",
        "in the same park near dusk with gentle haze",
        "in the same park after light rain, reflective ground",
        "in the same park at dawn with mild fog and warm rim light",
    )
    moods: tuple = (
        "calm cinematic realism, natural details",
        "cinematic realism, high detail, stable composition",
        "slightly energetic composition, crisp detail",
        "dramatic but realistic cinematic grade, controlled contrast",
    )


def _interp_index(pos_0_1: float, n: int):
    if n <= 1:
        return 0, 0, 0.0
    x = max(0.0, min(1.0, float(pos_0_1))) * (n - 1)
    lo = int(math.floor(x))
    hi = min(lo + 1, n - 1)
    w = x - lo
    return lo, hi, w


def build_discrete_prompt(
    prompt_level: int,
    frame: int,
    total_frames: int,
    beat_pulse: float,
    pitch: float,
    cfg: PromptBlendConfig = PromptBlendConfig(),
) -> str:
    subject = cfg.subjects[max(0, min(len(cfg.subjects) - 1, int(prompt_level)))]
    phase = int((4 * frame) / max(1, total_frames))
    phase = max(0, min(len(cfg.scenes) - 1, phase))
    scene = cfg.scenes[phase]

    if beat_pulse > 0.65:
        mood = cfg.moods[3]
    elif pitch > 0.66:
        mood = cfg.moods[2]
    elif pitch < 0.33:
        mood = cfg.moods[0]
    else:
        mood = cfg.moods[1]
    return f"{subject}, {scene}, {mood}"


def build_blended_prompt(
    prompt_drive: float,
    frame: int,
    total_frames: int,
    beat_pulse: float,
    pitch: float,
    cfg: PromptBlendConfig = PromptBlendConfig(),
) -> str:
    """
    Blend neighboring subject/scene concepts for smoother transitions.
    """
    s_lo, s_hi, s_w = _interp_index(prompt_drive, len(cfg.subjects))
    scene_pos = 0.0 if total_frames <= 1 else frame / float(total_frames - 1)
    c_lo, c_hi, c_w = _interp_index(scene_pos, len(cfg.scenes))

    if beat_pulse > 0.65:
        m_idx = 3
    elif pitch > 0.66:
        m_idx = 2
    elif pitch < 0.33:
        m_idx = 0
    else:
        m_idx = 1
    primary = f"{cfg.subjects[s_lo]}, {cfg.scenes[c_lo]}, {cfg.moods[m_idx]}"
    secondary = f"{cfg.subjects[s_hi]}, {cfg.scenes[c_hi]}, {cfg.moods[m_idx]}"

    blend_w = max(s_w, c_w)
    if blend_w < 0.05:
        return primary
    return (
        f"{primary}. Secondary influence ({blend_w:.2f}): {secondary}. "
        f"Keep the primary subject identity stable."
    )
