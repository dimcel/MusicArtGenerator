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
        "elderly violinist",
        "young street dancer",
        "female astronaut",
        "samurai warrior",
    )
    scenes: tuple = (
        "in an autumn park with cinematic soft light",
        "in a neon city plaza at blue dusk",
        "on an alien shoreline with bioluminescent mist",
        "in a dawn field with fog and volumetric rays",
    )
    moods: tuple = (
        "calm atmospheric composition, cinematic realism",
        "cinematic realism, high detail, stable composition",
        "energetic dynamic composition, crisp detail",
        "dramatic high-contrast action frame",
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class SubjectBlendState:
    from_idx: int
    to_idx: int
    weight: float
    in_transition: bool


class SubjectTransitionController:
    """
    Holds subject identity for a minimum number of frames and
    transitions with a crossfade window.
    """

    def __init__(self, num_subjects: int, hold_frames: int = 32, transition_frames: int = 16):
        self.num_subjects = max(1, int(num_subjects))
        self.hold_frames = max(1, int(hold_frames))
        self.transition_frames = max(1, int(transition_frames))

        self.initialized = False
        self.current_idx = 0
        self.target_idx = 0
        self.prev_idx = 0
        self.hold_until = 0
        self.in_transition = False
        self.transition_start = 0

    def _target_index(self, subject_drive: float) -> int:
        if self.num_subjects <= 1:
            return 0
        x = _clamp01(subject_drive) * (self.num_subjects - 1)
        return int(round(x))

    def step(self, frame: int, subject_drive: float) -> SubjectBlendState:
        target = self._target_index(subject_drive)

        if not self.initialized:
            self.initialized = True
            self.current_idx = target
            self.target_idx = target
            self.prev_idx = target
            self.hold_until = frame + self.hold_frames
            return SubjectBlendState(target, target, 0.0, False)

        if self.in_transition:
            w = (frame - self.transition_start) / float(self.transition_frames)
            if w >= 1.0:
                self.in_transition = False
                self.current_idx = self.target_idx
                self.prev_idx = self.current_idx
                self.hold_until = frame + self.hold_frames
                return SubjectBlendState(self.current_idx, self.current_idx, 0.0, False)
            return SubjectBlendState(self.prev_idx, self.target_idx, _clamp01(w), True)

        if target != self.current_idx and frame >= self.hold_until:
            self.in_transition = True
            self.transition_start = frame
            self.prev_idx = self.current_idx
            self.target_idx = target
            return SubjectBlendState(self.prev_idx, self.target_idx, 0.0, True)

        return SubjectBlendState(self.current_idx, self.current_idx, 0.0, False)


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
    cfg: PromptBlendConfig = None,
) -> str:
    if cfg is None:
        cfg = PromptBlendConfig()
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
    cfg: PromptBlendConfig = None,
    subject_blend: SubjectBlendState = None,
) -> str:
    """
    Blend neighboring subject/scene concepts for smoother transitions.
    """
    if cfg is None:
        cfg = PromptBlendConfig()

    if subject_blend is None:
        s_lo, s_hi, s_w = _interp_index(prompt_drive, len(cfg.subjects))
    else:
        s_lo, s_hi, s_w = (
            subject_blend.from_idx,
            subject_blend.to_idx,
            _clamp01(subject_blend.weight),
        )
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
