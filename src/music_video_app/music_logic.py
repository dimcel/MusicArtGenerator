from __future__ import annotations

"""Music-reactive helper logic for the video runner."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
try:
    from PIL import Image, ImageEnhance
except ModuleNotFoundError:
    Image = None
    ImageEnhance = None

from src.music_feature_mapper import MusicMappingConfig


def _load_and_resize_init_image(path: str, width: int, height: int) -> Image.Image:
    if Image is None:
        raise ImportError("Pillow is required for init-image loading/resizing.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Init image not found: {path}")
    image = Image.open(p).convert("RGB")
    if image.size != (width, height):
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
        image = image.resize((width, height), resample)
    return image


def _apply_identity_anchor(prompt: str, lock_identity: bool, identity_prompt: str) -> str:
    if not lock_identity:
        return prompt
    anchor = str(identity_prompt or "").strip()
    if not anchor:
        return prompt
    return f"{prompt}. Keep the same main subject identity: {anchor}."


def _parse_prompt_candidates(user_prompt: str):
    """
    Parse user prompt input as:
    - single prompt string
    - '||' separated prompt list
    - JSON list string, e.g. ["prompt A", "prompt B"]
    """
    raw = str(user_prompt or "").strip()
    if not raw:
        return ["cinematic music video frame"]

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if str(x).strip()]
                if items:
                    return items
        except json.JSONDecodeError:
            pass

    if "||" in raw:
        items = [x.strip() for x in raw.split("||") if x.strip()]
        if items:
            return items

    return [raw]


def _apply_music_color_fx(
    image: Image.Image,
    beat_pulse: float,
    energy: float,
    brightness: float,
) -> Image.Image:
    if Image is None or ImageEnhance is None:
        raise ImportError("Pillow is required for music color effects.")
    """
    Simple music color FX:
    - beat/energy -> saturation + contrast pop
    - brightness -> warm/cool tint direction
    """
    sat = 1.0 + 0.18 * float(beat_pulse) + 0.08 * (float(energy) - 0.5)
    sat = max(0.88, min(1.32, sat))
    contrast = 1.0 + 0.12 * float(beat_pulse) + 0.05 * (float(energy) - 0.5)
    contrast = max(0.90, min(1.25, contrast))

    out = ImageEnhance.Color(image).enhance(sat)
    out = ImageEnhance.Contrast(out).enhance(contrast)

    tint_color = (220, 235, 255) if float(brightness) >= 0.55 else (255, 235, 220)
    tint_alpha = 0.04 + 0.10 * float(beat_pulse)
    tint_alpha = max(0.02, min(0.14, tint_alpha))
    overlay = Image.new("RGB", out.size, tint_color)
    out = Image.blend(out, overlay, tint_alpha)
    return out


def _apply_onset_jitter_controls(
    controls: Dict[str, float],
    onset_value: float,
    frame: int,
    mapper_cfg: MusicMappingConfig,
    jitter_strength: float = 1.0,
) -> Dict[str, float]:
    """
    Small deterministic shutter/jitter from onset spikes.
    """
    strength = float(max(0.0, min(1.0, jitter_strength)))
    if strength <= 1e-6:
        return controls

    onset_v = float(max(0.0, min(1.0, onset_value)))
    if onset_v < 0.35:
        return controls

    amp = (onset_v - 0.35) / 0.65
    rng = np.random.default_rng(100000 + int(frame))

    jitter_px = (0.5 + 2.7 * amp) * strength
    jitter_ang = (0.10 + 0.90 * amp) * strength
    dx = float(rng.uniform(-1.0, 1.0)) * jitter_px
    dy = float(rng.uniform(-1.0, 1.0)) * jitter_px
    da = float(rng.uniform(-1.0, 1.0)) * jitter_ang

    out = dict(controls)
    out["tx_delta"] = float(out["tx_delta"] + dx)
    out["ty_delta"] = float(out["ty_delta"] + dy)
    out["angle_delta"] = float(out["angle_delta"] + da)

    pan_cap = float(mapper_cfg.pan_abs_max)
    ang_cap = float(mapper_cfg.angle_abs_max)
    out["tx_delta"] = max(-pan_cap, min(pan_cap, out["tx_delta"]))
    out["ty_delta"] = max(-pan_cap, min(pan_cap, out["ty_delta"]))
    out["angle_delta"] = max(-ang_cap, min(ang_cap, out["angle_delta"]))
    return out


def _re_anchor_profile(level: str):
    lv = str(level or "mid").strip().lower()
    if lv == "small":
        return 0.12, 0.46, 0.015, 1.00
    if lv == "big":
        return 0.24, 0.36, 0.006, 1.22
    if lv == "mid":
        return 0.18, 0.42, 0.01, 1.10
    try:
        alpha = float(lv)
    except ValueError:
        # Fallback to mid if input is neither a known preset nor a float.
        return 0.18, 0.42, 0.01, 1.10
    alpha = max(0.0, min(1.0, alpha))
    # Keep the rest of the profile stable (mid) and expose direct alpha control.
    return alpha, 0.42, 0.01, 1.10


def _build_stability_curve(
    onset_curve: np.ndarray,
    pitch_curve: np.ndarray,
    bright_curve: np.ndarray,
) -> np.ndarray:
    """
    Estimate melodic steadiness in [0, 1].
    High values mean:
    - low onset activity
    - low frame-to-frame pitch change
    - low frame-to-frame brightness change
    """
    if onset_curve.size == 0:
        return onset_curve

    pitch_delta = np.abs(np.diff(pitch_curve, prepend=float(pitch_curve[0])))
    bright_delta = np.abs(np.diff(bright_curve, prepend=float(bright_curve[0])))

    # Heavier onset penalty; pitch/brightness deltas capture "steady note" behavior.
    stability = 1.0 - (
        0.62 * onset_curve
        + 1.20 * pitch_delta
        + 0.55 * bright_delta
    )
    return np.clip(stability, 0.0, 1.0).astype(np.float32)


def _build_stable_run_lengths(
    stability_curve: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    For each frame, count consecutive frames ending at i with stability >= threshold.
    """
    out = np.zeros_like(stability_curve, dtype=np.int32)
    run = 0
    for i, v in enumerate(stability_curve):
        if float(v) >= threshold:
            run += 1
        else:
            run = 0
        out[i] = run
    return out


def _remove_short_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Remove active runs shorter than min_len from a boolean mask.
    """
    m = np.asarray(mask, dtype=bool).copy()
    if m.size == 0 or min_len <= 1:
        return m

    start = -1
    for i in range(m.size + 1):
        is_on = bool(m[i]) if i < m.size else False
        if is_on and start < 0:
            start = i
        if (not is_on) and start >= 0:
            if i - start < min_len:
                m[start:i] = False
            start = -1
    return m


def _extract_true_runs(mask: np.ndarray):
    """
    Return list of (start_inclusive, end_exclusive) for True runs.
    """
    m = np.asarray(mask, dtype=bool)
    runs = []
    start = -1
    for i in range(m.size + 1):
        is_on = bool(m[i]) if i < m.size else False
        if is_on and start < 0:
            start = i
        if (not is_on) and start >= 0:
            runs.append((start, i))
            start = -1
    return runs


def _build_auto_steady_activation_mask(
    stability_curve: np.ndarray,
    energy_curve: np.ndarray,
    onset_curve: np.ndarray,
    beat_frames: list,
    fps: int,
    activation_ratio: float = 1.0,
    shift_probability: float = 1.0,
):
    """
    Build an activation mask directly from music features.
    activation_ratio applies to eligible stable runs (not all frames).
    shift_probability is per-run keep probability after ratio selection.
    """
    stability = np.asarray(stability_curve, dtype=np.float32)
    energy = np.asarray(energy_curve, dtype=np.float32)
    onset = np.asarray(onset_curve, dtype=np.float32)

    if stability.size == 0:
        return np.zeros(0, dtype=bool), {
            "stable_threshold": 0.0,
            "activation_ratio": float(activation_ratio),
            "shift_probability": float(shift_probability),
            "eligible_frame_ratio": 0.0,
            "actual_frame_ratio": 0.0,
            "min_run_frames": 0,
            "beat_span_frames": 0,
            "energy_gate": 0.0,
            "eligible_runs": 0,
            "selected_runs": 0,
            "active_runs": 0,
        }

    # Require "present" signal energy so near-silent or very low-energy
    # stable regions don't trigger directional motion.
    energy_gate = float(max(0.08, np.quantile(energy, 0.35)))
    energy_ok = energy >= energy_gate

    stability_no_onset = np.clip(stability - 0.30 * onset, 0.0, 1.0)
    score = np.clip(stability_no_onset * np.asarray(energy_ok, dtype=np.float32), 0.0, 1.0)
    stable_threshold = float(max(0.55, np.quantile(stability_no_onset, 0.60)))

    if len(beat_frames) >= 2:
        beat_arr = np.asarray(sorted(beat_frames), dtype=np.int32)
        beat_gaps = np.diff(beat_arr)
        beat_span = int(np.median(beat_gaps)) if beat_gaps.size else max(2, int(0.5 * fps))
    else:
        beat_span = max(2, int(0.5 * fps))

    min_run_frames = max(3, int(round(0.45 * beat_span)))
    eligible_raw = np.logical_and(stability_no_onset >= stable_threshold, energy_ok)
    eligible_mask = _remove_short_true_runs(eligible_raw, min_len=min_run_frames)
    eligible_runs = _extract_true_runs(eligible_mask)
    eligible_frame_ratio = float(np.asarray(eligible_mask, dtype=np.float32).mean())

    activation_ratio = max(0.0, min(1.0, float(activation_ratio)))
    shift_probability = max(0.0, min(1.0, float(shift_probability)))

    selected_runs = []
    if eligible_runs and activation_ratio > 0.0:
        run_scores = []
        for idx, (a, b) in enumerate(eligible_runs):
            run_score = float(np.mean(score[a:b])) if b > a else 0.0
            run_scores.append((run_score, idx))

        run_scores.sort(reverse=True, key=lambda x: x[0])
        keep_count = int(round(activation_ratio * len(eligible_runs)))
        keep_count = min(len(eligible_runs), max(0, keep_count))
        if keep_count == 0 and activation_ratio > 0.0:
            keep_count = 1
        selected_indices = sorted(idx for _, idx in run_scores[:keep_count])
        selected_runs = [eligible_runs[i] for i in selected_indices]

    active_mask = np.zeros_like(eligible_mask, dtype=bool)
    if selected_runs and shift_probability > 0.0:
        rng = np.random.default_rng(42)
        for a, b in selected_runs:
            if float(rng.random()) <= shift_probability:
                active_mask[a:b] = True

    active_runs = _extract_true_runs(active_mask)
    actual_frame_ratio = float(np.asarray(active_mask, dtype=np.float32).mean())

    return active_mask.astype(bool), {
        "stable_threshold": stable_threshold,
        "activation_ratio": activation_ratio,
        "shift_probability": shift_probability,
        "eligible_frame_ratio": eligible_frame_ratio,
        "actual_frame_ratio": actual_frame_ratio,
        "min_run_frames": min_run_frames,
        "beat_span_frames": beat_span,
        "energy_gate": energy_gate,
        "eligible_runs": len(eligible_runs),
        "selected_runs": len(selected_runs),
        "active_runs": len(active_runs),
    }


def _build_quiet_hold_mask(
    energy_curve: np.ndarray,
    onset_curve: np.ndarray,
    fps: int,
):
    """
    Build a simple quiet-section mask from low energy + low onset runs.
    Internal thresholds are auto-derived; no user tuning args required.
    """
    energy = np.asarray(energy_curve, dtype=np.float32)
    onset = np.asarray(onset_curve, dtype=np.float32)

    if energy.size == 0 or onset.size == 0:
        return np.zeros(0, dtype=bool), {
            "energy_gate": 0.0,
            "onset_gate": 0.0,
            "min_run_frames": 0,
            "quiet_frame_ratio": 0.0,
            "quiet_runs": 0,
        }

    energy_gate = float(min(0.32, max(0.08, np.quantile(energy, 0.28))))
    onset_gate = float(min(0.30, max(0.08, np.quantile(onset, 0.35))))
    quiet_raw = np.logical_and(energy <= energy_gate, onset <= onset_gate)
    min_run_frames = max(3, int(round(0.35 * fps)))
    quiet_mask = _remove_short_true_runs(quiet_raw, min_len=min_run_frames)
    quiet_runs = _extract_true_runs(quiet_mask)

    return quiet_mask.astype(bool), {
        "energy_gate": energy_gate,
        "onset_gate": onset_gate,
        "min_run_frames": min_run_frames,
        "quiet_frame_ratio": float(np.asarray(quiet_mask, dtype=np.float32).mean()),
        "quiet_runs": len(quiet_runs),
    }


def _build_soft_run_gain(mask: np.ndarray, fade_frames: int) -> np.ndarray:
    """
    Build a 0..1 gain curve per True run with linear fade-in/out.
    """
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(m.shape, dtype=np.float32)
    if m.size == 0:
        return out

    fade = max(1, int(fade_frames))
    for a, b in _extract_true_runs(m):
        if b <= a:
            continue
        for i in range(a, b):
            d_in = i - a + 1
            d_out = b - i
            g_in = min(1.0, float(d_in) / float(fade))
            g_out = min(1.0, float(d_out) / float(fade))
            out[i] = float(min(g_in, g_out))
    return out


def _build_onset_twist_gain(
    onset_curve: np.ndarray,
    fps: int,
):
    """
    Build right-twist gain [0,1] from onset spikes with short decay.
    """
    onset = np.asarray(onset_curve, dtype=np.float32)
    if onset.size == 0:
        return np.zeros(0, dtype=np.float32), {
            "threshold": 0.0,
            "decay_frames": 0,
            "active_frame_ratio": 0.0,
        }

    threshold = float(max(0.32, np.quantile(onset, 0.70)))
    denom = max(1e-6, 1.0 - threshold)
    peak = np.clip((onset - threshold) / denom, 0.0, 1.0).astype(np.float32)

    decay_frames = max(2, int(round(0.22 * fps)))
    decay = float(np.exp(-1.0 / float(decay_frames)))
    gain = np.zeros_like(peak, dtype=np.float32)
    carry = 0.0
    for i, v in enumerate(peak):
        carry = max(float(v), carry * decay)
        gain[i] = float(carry)

    gain = np.clip(gain, 0.0, 1.0).astype(np.float32)
    active_frame_ratio = float(np.asarray(gain > 0.05, dtype=np.float32).mean())
    return gain, {
        "threshold": threshold,
        "decay_frames": decay_frames,
        "active_frame_ratio": active_frame_ratio,
    }


def _steady_direction_from_features(pitch: float, brightness: float):
    """
    Temporary debug behavior: force right-only shift.
    """
    _ = pitch, brightness
    return "right", 1.0, 0.0
