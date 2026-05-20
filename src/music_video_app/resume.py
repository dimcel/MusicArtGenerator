"""Resume/state helpers for the music video runner."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Optional

from src.prompt_blender import SubjectTransitionController


def _sanitize_label(value) -> str:
    s = str(value).strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s[:40] if len(s) > 40 else s


def _build_run_args(
    audio_path: str,
    fps: int,
    device: str,
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
    steady_twist: bool,
    steady_twist_max_deg: float,
    quiet_hold: bool,
) -> Dict[str, object]:
    _ = max_seconds
    init_image_abs = None
    if init_image:
        init_image_abs = str(Path(init_image).resolve())

    return {
        "audio_path": str(audio_path),
        "fps": int(fps),
        "device": str(device),
        "mode": str(mode),
        "cadence": int(cadence),
        "prompt_mode": str(prompt_mode),
        "coherence": str(coherence),
        "subject_hold_frames": int(subject_hold_frames),
        "subject_transition_frames": int(subject_transition_frames),
        "subject_smoothing_window": int(subject_smoothing_window),
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
        "steady_twist": bool(steady_twist),
        "steady_twist_max_deg": float(steady_twist_max_deg),
        "quiet_hold": bool(quiet_hold),
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
        loaded_states = []
        for p in candidates:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_state_file"] = str(p)
            loaded_states.append(data)

        def _rank(item: Dict[str, object]):
            state_file = Path(str(item["_state_file"]))
            return (
                int(item.get("last_completed_frame", -1)),
                float(state_file.stat().st_mtime),
            )

        best = max(loaded_states, key=_rank)
        best["_state_file_candidates"] = [Path(str(d["_state_file"])).name for d in loaded_states]
        return best
    return None


def _changed_run_arg_keys(previous: Dict[str, object], current: Dict[str, object]) -> list:
    changed = []
    for k in sorted(set(previous.keys()) | set(current.keys())):
        if previous.get(k) != current.get(k):
            changed.append(k)
    return changed


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
