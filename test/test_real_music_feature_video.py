"""
Real Music Feature -> Video Test (Simple)

Uses a real audio file:
1) Extract audio features (beat/energy/onset/brightness/pitch)
2) Map features to controls (strength/cfg/zoom/pan/angle/prompt)
3) Run Deforum-like loop: transform -> img2img
4) Export video and optionally mux original audio

Resume support:
- Pass --resume-dir to continue an interrupted run.
- Checkpoint is saved after each completed frame as:
  resume_state_<args_slug>.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from audio_feature_extractor import AudioFeatureExtractor
from cadence_coherence import CadenceCoherence
from color_coherence import apply_color_coherence
from controlnet_conditioning import build_canny_control_image
from image_generator import ImageGenerationConfig, ImageGenerator
from image_noise import add_gaussian_noise
from image_transform import transform_image
from music_feature_mapper import (
    MusicMappingConfig,
    calibrate_feature_curve,
    map_frame_to_controls,
)
from prompt_blender import (
    SubjectTransitionController,
)


def _sanitize_label(value) -> str:
    s = str(value).strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s[:40] if len(s) > 40 else s


def _build_run_args(
    audio_path: str,
    fps: int,
    mode: str,
    cadence: int,
    prompt_mode: str,
    coherence: str,
    subject_hold_frames: int,
    subject_transition_frames: int,
    subject_smoothing_window: int,
    max_seconds: float,
    use_controlnet: bool,
    controlnet_model: str,
    control_scale_base: float,
    control_scale_beat_boost: float,
    control_scale_onset_boost: float,
    zoom_base: float,
    zoom_beat_boost: float,
    canny_low: int,
    canny_high: int,
    init_image: Optional[str],
    concept_mode: str,
    identity_prompt: str,
    user_prompt: str,
    prompt_change_every_beats: int,
    music_color_fx: bool,
    onset_jitter: bool,
    color_coherence_mode: str,
    color_coherence_strength: float,
    re_anchor: bool,
    re_anchor_strength: str,
    re_anchor_every_frames: int,
    disable_camera_motion: bool,
    disable_music_change: bool,
    steady_shift: bool,
    steady_activation_mode: str,
    steady_activation_ratio: float,
    steady_shift_probability: float,
    steady_min_seconds: float,
    steady_threshold: float,
    steady_shift_pixels: float,
) -> Dict[str, object]:
    init_image_abs = None
    if init_image:
        init_image_abs = str(Path(init_image).resolve())

    return {
        "audio_path": str(audio_path),
        "fps": int(fps),
        "mode": str(mode),
        "cadence": int(cadence),
        "prompt_mode": str(prompt_mode),
        "coherence": str(coherence),
        "subject_hold_frames": int(subject_hold_frames),
        "subject_transition_frames": int(subject_transition_frames),
        "subject_smoothing_window": int(subject_smoothing_window),
        "max_seconds": float(max_seconds),
        "use_controlnet": bool(use_controlnet),
        "controlnet_model": str(controlnet_model),
        "control_scale_base": float(control_scale_base),
        "control_scale_beat_boost": float(control_scale_beat_boost),
        "control_scale_onset_boost": float(control_scale_onset_boost),
        "zoom_base": float(zoom_base),
        "zoom_beat_boost": float(zoom_beat_boost),
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "init_image": init_image_abs,
        "concept_mode": str(concept_mode),
        "identity_prompt": str(identity_prompt),
        "user_prompt": str(user_prompt),
        "prompt_change_every_beats": int(prompt_change_every_beats),
        "music_color_fx": bool(music_color_fx),
        "onset_jitter": bool(onset_jitter),
        "color_coherence_mode": str(color_coherence_mode),
        "color_coherence_strength": float(color_coherence_strength),
        "re_anchor": bool(re_anchor),
        "re_anchor_strength": str(re_anchor_strength),
        "re_anchor_every_frames": int(re_anchor_every_frames),
        "disable_camera_motion": bool(disable_camera_motion),
        "disable_music_change": bool(disable_music_change),
        "steady_shift": bool(steady_shift),
        "steady_activation_mode": str(steady_activation_mode),
        "steady_activation_ratio": float(steady_activation_ratio),
        "steady_shift_probability": float(steady_shift_probability),
        "steady_min_seconds": float(steady_min_seconds),
        "steady_threshold": float(steady_threshold),
        "steady_shift_pixels": float(steady_shift_pixels),
    }


def _build_args_slug(run_args: Dict[str, object]) -> str:
    digest = hashlib.sha1(
        json.dumps(run_args, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return (
        f"mode-{_sanitize_label(run_args['mode'])}"
        f"_pm-{_sanitize_label(run_args['prompt_mode'])}"
        f"_coh-{_sanitize_label(run_args['coherence'])}"
        f"_cn-{int(bool(run_args['use_controlnet']))}"
        f"_{digest}"
    )


def _state_path(output_dir: Path, args_slug: str) -> Path:
    return output_dir / f"resume_state_{args_slug}.json"


def _controller_to_dict(controller: SubjectTransitionController) -> Dict[str, object]:
    return {
        "initialized": bool(controller.initialized),
        "current_idx": int(controller.current_idx),
        "target_idx": int(controller.target_idx),
        "prev_idx": int(controller.prev_idx),
        "hold_until": int(controller.hold_until),
        "in_transition": bool(controller.in_transition),
        "transition_start": int(controller.transition_start),
    }


def _controller_from_dict(controller: SubjectTransitionController, state: Dict[str, object]):
    for name in (
        "initialized",
        "current_idx",
        "target_idx",
        "prev_idx",
        "hold_until",
        "in_transition",
        "transition_start",
    ):
        if name in state:
            setattr(controller, name, state[name])


def _save_resume_state(
    state_file: Path,
    run_args: Dict[str, object],
    args_slug: str,
    last_completed_frame: int,
    seed_state: float,
    subject_controller: SubjectTransitionController,
    total_frames: int,
    fps: int,
    prompt_state: Optional[Dict[str, int]],
    cli_args: Optional[list],
):
    payload = {
        "schema_version": 1,
        "args_slug": args_slug,
        "run_args": run_args,
        "cli_args": cli_args or [],
        "last_completed_frame": int(last_completed_frame),
        "seed_state": float(seed_state),
        "subject_controller": _controller_to_dict(subject_controller),
        "prompt_state": prompt_state or {},
        "total_frames": int(total_frames),
        "fps": int(fps),
    }
    tmp = state_file.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(state_file)


def _load_resume_state(
    output_dir: Path,
    args_slug: str,
) -> Optional[Dict[str, object]]:
    exact = _state_path(output_dir, args_slug)
    if exact.exists():
        with exact.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_state_file"] = str(exact)
        return data

    candidates = sorted(output_dir.glob("resume_state_*.json"))
    if len(candidates) == 1:
        with candidates[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_state_file"] = str(candidates[0])
        return data
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(
            "Multiple resume state files found and no exact args match. "
            f"Available: {names}"
        )
    return None


def _find_last_frame(output_dir: Path) -> int:
    max_frame = -1
    for p in output_dir.glob("frame_*.png"):
        m = re.match(r"frame_(\d+)\.png$", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > max_frame:
            max_frame = idx
    return max_frame


def _load_and_resize_init_image(path: str, width: int, height: int) -> Image.Image:
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
) -> Dict[str, float]:
    """
    Small deterministic shutter/jitter from onset spikes.
    """
    onset_v = float(max(0.0, min(1.0, onset_value)))
    if onset_v < 0.35:
        return controls

    amp = (onset_v - 0.35) / 0.65
    rng = np.random.default_rng(100000 + int(frame))

    jitter_px = 0.5 + 2.7 * amp
    jitter_ang = 0.10 + 0.90 * amp
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


def _steady_direction_from_features(pitch: float, brightness: float):
    """
    Temporary debug behavior: force right-only shift.
    """
    _ = pitch, brightness
    return "right", 1.0, 0.0


def run(
    audio_path: str,
    fps: int = 24,
    mode: str = "full",
    cadence: int = 3,
    prompt_mode: str = "blend",
    coherence: str = "blend",
    subject_hold_frames: int = 32,
    subject_transition_frames: int = 16,
    subject_smoothing_window: int = 41,
    max_seconds: float = 12.0,
    with_audio: bool = True,
    use_controlnet: bool = False,
    controlnet_model: str = "lllyasviel/sd-controlnet-canny",
    control_scale_base: float = 0.80,
    control_scale_beat_boost: float = 0.30,
    control_scale_onset_boost: float = 0.20,
    zoom_base: float = 1.002,
    zoom_beat_boost: float = 0.020,
    canny_low: int = 100,
    canny_high: int = 200,
    init_image: Optional[str] = None,
    concept_mode: str = "identity",
    identity_prompt: str = "",
    user_prompt: str = "",
    prompt_change_every_beats: int = 1,
    music_color_fx: bool = False,
    onset_jitter: bool = False,
    color_coherence_mode: str = "none",
    color_coherence_strength: float = 0.60,
    re_anchor: bool = False,
    re_anchor_strength: str = "mid",
    re_anchor_every_frames: int = 12,
    disable_camera_motion: bool = False,
    disable_music_change: bool = False,
    steady_shift: bool = False,
    steady_activation_mode: str = "auto",
    steady_activation_ratio: float = 1.0,
    steady_shift_probability: float = 1.0,
    steady_min_seconds: float = 1.0,
    steady_threshold: float = 0.72,
    steady_shift_pixels: float = 6.0,
    resume_dir: Optional[str] = None,
    cli_args: Optional[list] = None,
):
    extractor = AudioFeatureExtractor(audio_path=audio_path, fps=fps).load()

    if max_seconds is not None and max_seconds > 0:
        total_frames = int(min(extractor.duration_seconds, max_seconds) * fps)
    else:
        total_frames = int(extractor.duration_seconds * fps)
    total_frames = max(2, total_frames)

    features = extractor.extract(total_frames=total_frames)
    beat_set = set(features.beat_frames)
    mapper_cfg = MusicMappingConfig(
        # Test profile: camera motion driven by zoom only (energy + beat pulse).
        zoom_base=float(zoom_base),
        zoom_energy_boost=0.008,
        zoom_beat_boost=float(zoom_beat_boost),
        zoom_min=0.995,
        zoom_max=1.045,
        # Disable pan/rotation camera movement for isolation tests.
        pan_base=0.0,
        pan_onset_boost=0.0,
        pan_wave_amp=0.0,
        angle_base=0.0,
        angle_onset_boost=0.0,
        angle_wave_amp=0.0,
        control_scale_base=float(control_scale_base),
        control_scale_beat_boost=float(control_scale_beat_boost),
        control_scale_onset_boost=float(control_scale_onset_boost),
    )

    # Per-track robust normalization (simple auto-calibration).
    energy_curve = calibrate_feature_curve(features.energy, low_q=0.05, high_q=0.98, gamma=1.0)
    onset_curve = calibrate_feature_curve(features.onset, low_q=0.20, high_q=0.995, gamma=1.15)
    bright_curve = calibrate_feature_curve(features.brightness, low_q=0.05, high_q=0.98, gamma=1.0)
    pitch_curve = calibrate_feature_curve(features.pitch, low_q=0.10, high_q=0.95, gamma=1.0)
    prompt_candidates = _parse_prompt_candidates(user_prompt)
    prompt_change_every_beats = max(1, int(prompt_change_every_beats))
    prompt_reactive_enabled = len(prompt_candidates) > 1 and (not disable_music_change)

    steady_activation_mode = str(steady_activation_mode or "auto").strip().lower()
    if steady_activation_mode not in ("auto", "manual"):
        raise ValueError("steady_activation_mode must be 'auto' or 'manual'")

    steady_activation_ratio = max(0.0, min(1.0, float(steady_activation_ratio)))
    steady_shift_probability = max(0.0, min(1.0, float(steady_shift_probability)))
    steady_threshold = max(0.0, min(1.0, float(steady_threshold)))
    steady_shift_pixels = abs(float(steady_shift_pixels))
    steady_min_frames = max(1, int(round(max(0.05, float(steady_min_seconds)) * fps)))
    stability_curve = _build_stability_curve(onset_curve, pitch_curve, bright_curve)

    stable_run_lengths = None
    steady_active_mask = None
    steady_auto_meta = None
    if steady_activation_mode == "manual":
        stable_run_lengths = _build_stable_run_lengths(stability_curve, threshold=steady_threshold)
    else:
        steady_active_mask, steady_auto_meta = _build_auto_steady_activation_mask(
            stability_curve=stability_curve,
            energy_curve=energy_curve,
            onset_curve=onset_curve,
            beat_frames=features.beat_frames,
            fps=fps,
            activation_ratio=steady_activation_ratio,
            shift_probability=steady_shift_probability,
        )

    run_args = _build_run_args(
        audio_path=audio_path,
        fps=fps,
        mode=mode,
        cadence=cadence,
        prompt_mode=prompt_mode,
        coherence=coherence,
        subject_hold_frames=subject_hold_frames,
        subject_transition_frames=subject_transition_frames,
        subject_smoothing_window=subject_smoothing_window,
        max_seconds=max_seconds,
        use_controlnet=use_controlnet,
        controlnet_model=controlnet_model,
        control_scale_base=control_scale_base,
        control_scale_beat_boost=control_scale_beat_boost,
        control_scale_onset_boost=control_scale_onset_boost,
        zoom_base=zoom_base,
        zoom_beat_boost=zoom_beat_boost,
        canny_low=canny_low,
        canny_high=canny_high,
        init_image=init_image,
        concept_mode=concept_mode,
        identity_prompt=identity_prompt,
        user_prompt=user_prompt,
        prompt_change_every_beats=prompt_change_every_beats,
        music_color_fx=music_color_fx,
        onset_jitter=onset_jitter,
        color_coherence_mode=color_coherence_mode,
        color_coherence_strength=color_coherence_strength,
        re_anchor=re_anchor,
        re_anchor_strength=re_anchor_strength,
        re_anchor_every_frames=re_anchor_every_frames,
        disable_camera_motion=disable_camera_motion,
        disable_music_change=disable_music_change,
        steady_shift=steady_shift,
        steady_activation_mode=steady_activation_mode,
        steady_activation_ratio=steady_activation_ratio,
        steady_shift_probability=steady_shift_probability,
        steady_min_seconds=steady_min_seconds,
        steady_threshold=steady_threshold,
        steady_shift_pixels=steady_shift_pixels,
    )
    args_slug = _build_args_slug(run_args)
    lock_identity = bool(init_image) and concept_mode == "identity"
    width, height = 512, 512

    if resume_dir:
        output_dir = Path(resume_dir)
    else:
        safe_name = Path(audio_path).stem.replace(" ", "_")
        output_dir = Path(f"real_music_feature_video_{safe_name}_{mode}_{total_frames}f")
    output_dir.mkdir(exist_ok=True)
    resume_state_file = _state_path(output_dir, args_slug)

    print(f"Audio: {audio_path}")
    print(f"Duration used: {total_frames / fps:.2f}s ({total_frames} frames @ {fps}fps)")
    print(f"BPM: {features.bpm:.1f}, beats: {len(features.beat_frames)}")
    print(f"Mode: {mode}, cadence: {cadence}, prompt_mode: {prompt_mode}, coherence: {coherence}")
    print(
        f"ControlNet: {'ON' if use_controlnet else 'OFF'}"
        + (
            f" (model={controlnet_model}, canny={canny_low}:{canny_high})"
            if use_controlnet
            else ""
        )
    )
    if init_image:
        print(f"Init image: {init_image} (concept_mode={concept_mode})")
    else:
        print("Init image: OFF")
    print(f"Music color FX: {'ON' if music_color_fx else 'OFF'}")
    print(f"Onset jitter: {'ON' if onset_jitter else 'OFF'}")
    print(f"Color coherence: {color_coherence_mode} (strength={color_coherence_strength:.2f})")
    print(
        f"Re-anchor: {'ON' if re_anchor else 'OFF'} "
        f"(strength={re_anchor_strength}, every={max(1, int(re_anchor_every_frames))}f)"
    )
    if steady_shift:
        if steady_activation_mode == "auto" and steady_auto_meta is not None:
            print(
                "Steady shift: ON "
                f"(mode=auto, run_ratio={steady_auto_meta['activation_ratio']:.2f}, "
                f"run_prob={steady_auto_meta['shift_probability']:.2f}, "
                f"eligible_runs={steady_auto_meta['eligible_runs']}, "
                f"active_runs={steady_auto_meta['active_runs']}, "
                f"eligible_f={steady_auto_meta['eligible_frame_ratio']:.2f}, "
                f"active_f={steady_auto_meta['actual_frame_ratio']:.2f}, "
                f"sth={steady_auto_meta['stable_threshold']:.2f}, "
                f"eg={steady_auto_meta['energy_gate']:.2f}, "
                f"min_run={steady_auto_meta['min_run_frames']}f, "
                f"px={steady_shift_pixels:.2f})"
            )
        else:
            print(
                "Steady shift: ON "
                f"(mode=manual, min={steady_min_seconds:.2f}s/{steady_min_frames}f, "
                f"thr={steady_threshold:.2f}, px={steady_shift_pixels:.2f})"
            )
    else:
        print("Steady shift: OFF")
    print(f"Camera motion: {'OFF' if disable_camera_motion else 'ON'}")
    if len(prompt_candidates) > 1:
        print(
            "User prompts: "
            f"{len(prompt_candidates)} items "
            f"(reactive={'ON' if prompt_reactive_enabled else 'OFF'}, "
            f"change_every={prompt_change_every_beats} beat(s))"
        )
    elif str(user_prompt).strip():
        print(f"User prompt: {prompt_candidates[0]}")
    print(f"Output: {output_dir}/")
    print(f"Resume state: {resume_state_file.name}")

    generator = ImageGenerator(
        ImageGenerationConfig(
            width=width,
            height=height,
            enable_controlnet=use_controlnet,
            controlnet_type="canny",
            controlnet_model_id=controlnet_model,
        )
    )
    coherence_helper = CadenceCoherence(method=coherence, blend_alpha=0.35)
    subject_controller = SubjectTransitionController(
        num_subjects=4,
        hold_frames=subject_hold_frames,
        transition_frames=subject_transition_frames,
    )
    seed_state = 42.0
    re_anchor_every = max(1, int(re_anchor_every_frames))
    re_anchor_alpha, re_anchor_strength_cap, re_anchor_noise_cap, re_anchor_control_min = _re_anchor_profile(
        re_anchor_strength
    )

    current = None
    reference_frame = None
    start_frame = 0
    active_prompt_idx = 0
    beat_events_seen_for_prompt = 0

    if resume_dir:
        loaded = _load_resume_state(output_dir, args_slug)
        if loaded is not None:
            loaded_run_args = loaded.get("run_args")
            if loaded_run_args:
                loaded_run_args_cmp = dict(loaded_run_args)
                # Backward compatibility with resume states created
                # before steady-shift options existed.
                for k in (
                    "steady_shift",
                    "steady_activation_mode",
                    "steady_activation_ratio",
                    "steady_shift_probability",
                    "music_color_fx",
                    "onset_jitter",
                    "steady_min_seconds",
                    "steady_threshold",
                    "steady_shift_pixels",
                ):
                    if k not in loaded_run_args_cmp and k in run_args:
                        loaded_run_args_cmp[k] = run_args[k]
            else:
                loaded_run_args_cmp = None

            if loaded_run_args_cmp and loaded_run_args_cmp != run_args:
                raise RuntimeError(
                    "Resume state exists but run arguments differ from current command. "
                    "Use the same arguments, or choose a different resume directory."
                )
            last_frame = int(loaded.get("last_completed_frame", -1))
            frame_path = output_dir / f"frame_{last_frame:05d}.png"
            if not frame_path.exists():
                raise RuntimeError(
                    f"Resume state points to missing frame: {frame_path}"
                )
            current = Image.open(frame_path).convert("RGB")
            start_frame = last_frame + 1
            seed_state = float(loaded.get("seed_state", 42.0))
            _controller_from_dict(
                subject_controller,
                loaded.get("subject_controller", {}),
            )
            loaded_prompt_state = loaded.get("prompt_state", {})
            if isinstance(loaded_prompt_state, dict) and loaded_prompt_state:
                active_prompt_idx = int(loaded_prompt_state.get("active_prompt_idx", 0))
                beat_events_seen_for_prompt = int(
                    loaded_prompt_state.get("beat_events_seen_for_prompt", 0)
                )
            else:
                # Backward compatibility with old resume files:
                # reconstruct prompt progression from completed frames.
                if prompt_reactive_enabled:
                    beat_events_seen_for_prompt = sum(
                        1 for b in features.beat_frames if 1 <= int(b) <= last_frame
                    )
                    active_prompt_idx = (
                        beat_events_seen_for_prompt // prompt_change_every_beats
                    ) % max(1, len(prompt_candidates))
                else:
                    beat_events_seen_for_prompt = 0
                    active_prompt_idx = 0

            if prompt_candidates:
                active_prompt_idx %= len(prompt_candidates)
            print(
                f"Resuming from frame {start_frame} using state "
                f"{Path(loaded['_state_file']).name}"
            )
            frame0_path = output_dir / "frame_00000.png"
            if frame0_path.exists():
                reference_frame = Image.open(frame0_path).convert("RGB")
        else:
            last_frame = _find_last_frame(output_dir)
            if last_frame >= 0:
                raise RuntimeError(
                    "Found frames in resume dir but no resume state file for current args. "
                    "Use the same args as before, or clear/change resume directory."
                )
            print("No resume state found. Starting from frame 0.")

    if start_frame == 0:
        if init_image:
            current = _load_and_resize_init_image(init_image, width=width, height=height)
            print("Loaded init image as frame 0.")
        else:
            first_prompt = prompt_candidates[active_prompt_idx]
            current = generator.generate_from_text(prompt=first_prompt, seed=42)
        reference_frame = current.copy()
        current.save(output_dir / "frame_00000.png")
        _save_resume_state(
            state_file=resume_state_file,
            run_args=run_args,
            args_slug=args_slug,
            last_completed_frame=0,
            seed_state=seed_state,
            subject_controller=subject_controller,
            total_frames=total_frames,
            fps=fps,
            prompt_state={
                "active_prompt_idx": int(active_prompt_idx),
                "beat_events_seen_for_prompt": int(beat_events_seen_for_prompt),
            },
            cli_args=cli_args,
        )
        start_frame = 1

    if not prompt_reactive_enabled:
        active_prompt_idx = 0
        beat_events_seen_for_prompt = 0

    if start_frame >= total_frames:
        print("All frames already generated; skipping frame generation.")

    steady_prev_active = False
    steady_dir_label = "none"
    steady_dir_x = 0.0
    steady_dir_y = 0.0

    for frame in range(start_frame, total_frames):
        if reference_frame is None:
            frame0_path = output_dir / "frame_00000.png"
            if frame0_path.exists():
                reference_frame = Image.open(frame0_path).convert("RGB")

        controls = map_frame_to_controls(
            frame=frame,
            total_frames=total_frames,
            energy=float(energy_curve[frame]),
            onset=float(onset_curve[frame]),
            brightness=float(bright_curve[frame]),
            pitch=float(pitch_curve[frame]),
            beat_pulse=float(features.beat_pulse[frame]),
            cfg=mapper_cfg,
        )

        if disable_music_change:
            controls = {
                "strength": float(mapper_cfg.strength_base),
                "cfg_scale": float(mapper_cfg.cfg_base),
                "zoom_delta": float(mapper_cfg.zoom_base),
                "tx_delta": float(mapper_cfg.pan_base),
                "ty_delta": float(0.35 * mapper_cfg.pan_base),
                "angle_delta": float(mapper_cfg.angle_base),
                "noise_amount": float(mapper_cfg.noise_base),
                "control_scale": float(mapper_cfg.control_scale_base),
                "prompt_drive": 0.5,
                "prompt_level": 2,
                "seed_jump": 0,
            }

        steady_active = False
        steady_direction = "none"
        if steady_shift and not disable_music_change:
            if steady_activation_mode == "auto":
                steady_active = bool(steady_active_mask[frame]) if steady_active_mask is not None else False
            else:
                steady_active = (
                    bool(stable_run_lengths is not None)
                    and int(stable_run_lengths[frame]) >= steady_min_frames
                )

            if steady_active:
                if not steady_prev_active:
                    steady_dir_label, steady_dir_x, steady_dir_y = _steady_direction_from_features(
                        pitch=float(pitch_curve[frame]),
                        brightness=float(bright_curve[frame]),
                    )
                steady_direction = steady_dir_label
                stability_gain = float(stability_curve[frame])
                shift_px = steady_shift_pixels * (0.65 + 0.55 * stability_gain)

                controls["tx_delta"] = float(controls["tx_delta"] + steady_dir_x * shift_px)
                controls["ty_delta"] = float(controls["ty_delta"] + steady_dir_y * shift_px)

                pan_cap = max(float(mapper_cfg.pan_abs_max), steady_shift_pixels * 2.0)
                controls["tx_delta"] = max(-pan_cap, min(pan_cap, float(controls["tx_delta"])))
                controls["ty_delta"] = max(-pan_cap, min(pan_cap, float(controls["ty_delta"])))
            else:
                steady_dir_label = "none"
                steady_dir_x = 0.0
                steady_dir_y = 0.0
        else:
            steady_dir_label = "none"
            steady_dir_x = 0.0
            steady_dir_y = 0.0

        steady_prev_active = bool(steady_active)
        onset_jitter_active = 0
        if onset_jitter and not disable_music_change:
            if float(onset_curve[frame]) >= 0.35:
                onset_jitter_active = 1
            controls = _apply_onset_jitter_controls(
                controls=controls,
                onset_value=float(onset_curve[frame]),
                frame=frame,
                mapper_cfg=mapper_cfg,
            )

        if disable_camera_motion:
            transformed = current.copy()
        else:
            transformed = transform_image(
                current,
                zoom=controls["zoom_delta"],
                angle=controls["angle_delta"],
                translation_x=controls["tx_delta"],
                translation_y=controls["ty_delta"],
            )

        prompt_switched = 0
        if prompt_reactive_enabled and (frame in beat_set):
            beat_events_seen_for_prompt += 1
            if beat_events_seen_for_prompt % prompt_change_every_beats == 0:
                prev_prompt_idx = active_prompt_idx
                active_prompt_idx = (active_prompt_idx + 1) % len(prompt_candidates)
                prompt_switched = int(active_prompt_idx != prev_prompt_idx)

        prompt = prompt_candidates[active_prompt_idx]
        transition_active = False
        prompt = _apply_identity_anchor(prompt, lock_identity=lock_identity, identity_prompt=identity_prompt)

        # Adaptive cadence:
        # always diffuse during subject transitions, otherwise cadence+beats.
        do_diffuse = (
            (mode == "full")
            or transition_active
            or (frame in beat_set)
            or (frame % max(1, cadence) == 0)
        )

        used_strength = float(controls["strength"])
        used_cfg = float(controls["cfg_scale"])
        used_noise = float(controls["noise_amount"])
        used_control_scale = float(controls["control_scale"])
        used_seed = 42 + frame * 97 + int(controls["seed_jump"])
        re_anchor_applied = False

        if do_diffuse:
            target_seed = float(42 + frame * 97 + int(controls["seed_jump"]))
            # Seed travel-style interpolation during subject transitions.
            if transition_active:
                seed_state = seed_state + 0.25 * (target_seed - seed_state)
            else:
                seed_state = target_seed
            used_seed = int(seed_state)

            # Keep transitions stable: moderate strength, avoid spikes.
            if transition_active:
                used_strength = max(0.55, min(0.70, used_strength))
                used_cfg = min(used_cfg, 9.8)
                used_noise = min(used_noise, 0.05)

            # Init-image identity mode should preserve structure/colors aggressively.
            if lock_identity:
                used_seed = 42
                used_strength = max(0.30, min(0.50, used_strength))
                used_cfg = min(used_cfg, 8.5)
                used_noise = min(used_noise, 0.02)
                used_control_scale = max(0.90, used_control_scale)

            control_image = None
            if use_controlnet:
                control_image = build_canny_control_image(
                    transformed,
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                )

            noised = add_gaussian_noise(transformed, amount=used_noise, seed=used_seed + 17)

            # Periodically re-anchor to warped frame0 when init-image is provided.
            # This works for both concept modes (identity/transform).
            if bool(init_image) and re_anchor and reference_frame is not None:
                if frame % re_anchor_every == 0:
                    ref_warped = transform_image(
                        reference_frame,
                        zoom=controls["zoom_delta"],
                        angle=controls["angle_delta"],
                        translation_x=controls["tx_delta"],
                        translation_y=controls["ty_delta"],
                    )
                    noised = Image.blend(noised, ref_warped, re_anchor_alpha)
                    used_strength = min(used_strength, re_anchor_strength_cap)
                    used_noise = min(used_noise, re_anchor_noise_cap)
                    used_control_scale = max(used_control_scale, re_anchor_control_min)
                    re_anchor_applied = True

            current = generator.generate_from_image(
                init_image=noised,
                prompt=prompt,
                strength=used_strength,
                guidance_scale=used_cfg,
                seed=used_seed,
                control_image=control_image,
                controlnet_conditioning_scale=used_control_scale,
            )
        else:
            current = coherence_helper.apply(prev_frame=current, transformed_frame=transformed)

        if color_coherence_mode != "none" and reference_frame is not None:
            current = apply_color_coherence(
                method=color_coherence_mode,
                reference_frame=reference_frame,
                current_frame=current,
                strength=color_coherence_strength,
            )

        if music_color_fx and not disable_music_change:
            current = _apply_music_color_fx(
                current,
                beat_pulse=float(features.beat_pulse[frame]),
                energy=float(energy_curve[frame]),
                brightness=float(bright_curve[frame]),
            )

        current.save(output_dir / f"frame_{frame:05d}.png")
        _save_resume_state(
            state_file=resume_state_file,
            run_args=run_args,
            args_slug=args_slug,
            last_completed_frame=frame,
            seed_state=seed_state,
            subject_controller=subject_controller,
            total_frames=total_frames,
            fps=fps,
            prompt_state={
                "active_prompt_idx": int(active_prompt_idx),
                "beat_events_seen_for_prompt": int(beat_events_seen_for_prompt),
            },
            cli_args=cli_args,
        )

        if (frame + 1) % 20 == 0 or frame == total_frames - 1:
            beat_tag = "BEAT" if frame in beat_set else "-"
            print(
                f"{frame:>4}/{total_frames-1} [{beat_tag}] "
                f"eng={energy_curve[frame]:.2f} onset={onset_curve[frame]:.2f} pitch={pitch_curve[frame]:.2f} "
                f"str={used_strength:.3f} cfg={used_cfg:.2f} "
                f"zoom={controls['zoom_delta']:.4f} pan=({controls['tx_delta']:+.2f},{controls['ty_delta']:+.2f}) "
                f"stab={stability_curve[frame]:.2f} sact={int(steady_active)} sdir={steady_direction} "
                f"jit={onset_jitter_active} "
                f"pidx={active_prompt_idx} psw={prompt_switched} "
                f"noise={used_noise:.3f} trans={int(transition_active)} "
                f"cscale={used_control_scale:.3f} cn={'ON' if use_controlnet else 'OFF'} "
                f"ra={int(re_anchor_applied)}"
            )

    from frames_to_video import frames_to_video

    video_path = output_dir / f"real_music_feature_video_{mode}_{total_frames}f.mp4"
    frames_to_video(str(output_dir), str(video_path), fps=fps)
    print(f"Video: {video_path}")

    if with_audio:
        muxed_path = output_dir / f"real_music_feature_video_{mode}_{total_frames}f_with_audio.mp4"
        cmd = (
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -shortest "{muxed_path}"'
        )
        if os.system(cmd) == 0:
            print(f"Muxed: {muxed_path}")
        else:
            print("ffmpeg mux failed; silent video is still saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (mp3/wav/etc.)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--mode", choices=["full", "cadence"], default="full")
    parser.add_argument("--cadence", type=int, default=3)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help=(
            "User prompt override. Supports single prompt or prompt list with '||' "
            "separator (or JSON list string). In init-image identity mode, this is "
            "the main prompt source."
        ),
    )
    parser.add_argument(
        "--prompt-change-every-beats",
        type=int,
        default=1,
        help="When using prompt lists, switch to next prompt every N detected beats.",
    )
    parser.add_argument(
        "--music-color-fx",
        action="store_true",
        help="Enable simple music-driven color FX (beat pop + warm/cool tint).",
    )
    parser.add_argument(
        "--onset-jitter",
        action="store_true",
        help="Enable small shutter/jitter camera shake from onset spikes.",
    )
    parser.add_argument(
        "--color-coherence",
        choices=["none", "match_frame0_lab"],
        default="none",
        help="Optional color coherence post-process.",
    )
    parser.add_argument(
        "--color-coherence-strength",
        type=float,
        default=0.60,
        help="Blend strength for color coherence (0..1).",
    )
    parser.add_argument("--prompt-mode", choices=["blend", "hard"], default="blend")
    parser.add_argument("--coherence", choices=["none", "blend", "optical_flow", "rife", "film"], default="blend")
    parser.add_argument("--subject-hold-frames", type=int, default=32)
    parser.add_argument("--subject-transition-frames", type=int, default=16)
    parser.add_argument("--subject-smoothing-window", type=int, default=41)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=12.0,
        help="Limit runtime by using only first N seconds. <=0 means full audio.",
    )
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--controlnet", action="store_true", help="Enable ControlNet img2img (canny).")
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="ControlNet model id.",
    )
    parser.add_argument("--control-scale-base", type=float, default=0.80)
    parser.add_argument("--control-scale-beat-boost", type=float, default=0.30)
    parser.add_argument("--control-scale-onset-boost", type=float, default=0.20)
    parser.add_argument("--zoom-base", type=float, default=1.002, help="Base per-frame zoom multiplier for test profile.")
    parser.add_argument("--zoom-beat-boost", type=float, default=0.020, help="Additional zoom multiplier from beat pulse for test profile.")
    parser.add_argument(
        "--steady-shift",
        action="store_true",
        help="Enable aggressive pan direction during musically stable sections.",
    )
    parser.add_argument(
        "--steady-activation-mode",
        choices=["auto", "manual"],
        default="auto",
        help="auto=derive activation duration from music features, manual=use threshold+min-seconds.",
    )
    parser.add_argument(
        "--steady-activation-ratio",
        type=float,
        default=1.0,
        help="Auto mode: fraction of eligible music-driven shift runs to keep (0..1).",
    )
    parser.add_argument(
        "--steady-shift-probability",
        type=float,
        default=1.0,
        help="Auto mode: per-selected-run probability that shift is applied (0..1).",
    )
    parser.add_argument(
        "--steady-min-seconds",
        type=float,
        default=1.0,
        help="Manual mode only: minimum consecutive stable duration before shift activates.",
    )
    parser.add_argument(
        "--steady-threshold",
        type=float,
        default=0.72,
        help="Manual mode only: stability threshold in [0,1] for steady-shift activation.",
    )
    parser.add_argument(
        "--steady-shift-pixels",
        type=float,
        default=6.0,
        help="Base pan magnitude (pixels/frame) while steady-shift is active.",
    )
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument(
        "--init-image",
        type=str,
        default=None,
        help="Path to an initial image (jpg/png/webp). If set, frame 0 starts from this image.",
    )
    parser.add_argument(
        "--concept-mode",
        choices=["identity", "transform"],
        default="identity",
        help="How to treat init-image concept. identity=lock subject, transform=allow normal subject drift.",
    )
    parser.add_argument(
        "--identity-prompt",
        type=str,
        default="",
        help="Optional identity anchor text appended in identity concept mode.",
    )
    parser.add_argument(
        "--re-anchor",
        action="store_true",
        help="Periodically re-anchor diffusion to transformed frame 0 when --init-image is provided.",
    )
    parser.add_argument(
        "--re-anchor-strength",
        type=str,
        default="mid",
        help=(
            "Re-anchor strength preset (small|mid|big) or numeric blend alpha in [0,1] "
            "(e.g. 0.35)."
        ),
    )
    parser.add_argument(
        "--re-anchor-every-frames",
        type=int,
        default=12,
        help="Apply re-anchor every N diffusion frames.",
    )
    parser.add_argument(
        "--disable-camera-motion",
        action="store_true",
        help="Disable zoom/pan/rotation transform and keep camera static.",
    )
    parser.add_argument(
        "--disable-music-change",
        action="store_true",
        help="Disable music-driven control changes and use fixed controls each frame.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help=(
            "Resume an interrupted run from this directory. "
            "The script loads frame_*.png and resume_state_<args_slug>.json."
        ),
    )
    args = parser.parse_args()

    run(
        audio_path=args.audio,
        fps=args.fps,
        mode=args.mode,
        cadence=args.cadence,
        user_prompt=args.prompt,
        prompt_change_every_beats=args.prompt_change_every_beats,
        music_color_fx=args.music_color_fx,
        onset_jitter=args.onset_jitter,
        color_coherence_mode=args.color_coherence,
        color_coherence_strength=args.color_coherence_strength,
        prompt_mode=args.prompt_mode,
        coherence=args.coherence,
        subject_hold_frames=args.subject_hold_frames,
        subject_transition_frames=args.subject_transition_frames,
        subject_smoothing_window=args.subject_smoothing_window,
        max_seconds=args.max_seconds,
        with_audio=args.with_audio,
        use_controlnet=args.controlnet,
        controlnet_model=args.controlnet_model,
        control_scale_base=args.control_scale_base,
        control_scale_beat_boost=args.control_scale_beat_boost,
        control_scale_onset_boost=args.control_scale_onset_boost,
        zoom_base=args.zoom_base,
        zoom_beat_boost=args.zoom_beat_boost,
        steady_shift=args.steady_shift,
        steady_activation_mode=args.steady_activation_mode,
        steady_activation_ratio=args.steady_activation_ratio,
        steady_shift_probability=args.steady_shift_probability,
        steady_min_seconds=args.steady_min_seconds,
        steady_threshold=args.steady_threshold,
        steady_shift_pixels=args.steady_shift_pixels,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        init_image=args.init_image,
        concept_mode=args.concept_mode,
        identity_prompt=args.identity_prompt,
        re_anchor=args.re_anchor,
        re_anchor_strength=args.re_anchor_strength,
        re_anchor_every_frames=args.re_anchor_every_frames,
        disable_camera_motion=args.disable_camera_motion,
        disable_music_change=args.disable_music_change,
        resume_dir=args.resume_dir,
        cli_args=sys.argv[1:],
    )
