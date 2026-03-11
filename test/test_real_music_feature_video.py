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

from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from audio_feature_extractor import AudioFeatureExtractor
from cadence_coherence import CadenceCoherence
from controlnet_conditioning import build_canny_control_image
from image_generator import ImageGenerationConfig, ImageGenerator
from image_noise import add_gaussian_noise
from image_transform import transform_image
from music_feature_mapper import (
    MusicMappingConfig,
    build_subject_drive_curve,
    calibrate_feature_curve,
    map_frame_to_controls,
)
from prompt_blender import (
    SubjectTransitionController,
    build_blended_prompt,
    build_discrete_prompt,
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
    canny_low: int,
    canny_high: int,
) -> Dict[str, object]:
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
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
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
    canny_low: int = 100,
    canny_high: int = 200,
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
        control_scale_base=float(control_scale_base),
        control_scale_beat_boost=float(control_scale_beat_boost),
        control_scale_onset_boost=float(control_scale_onset_boost),
    )

    # Per-track robust normalization (simple auto-calibration).
    energy_curve = calibrate_feature_curve(features.energy, low_q=0.05, high_q=0.98, gamma=1.0)
    onset_curve = calibrate_feature_curve(features.onset, low_q=0.20, high_q=0.995, gamma=1.15)
    bright_curve = calibrate_feature_curve(features.brightness, low_q=0.05, high_q=0.98, gamma=1.0)
    pitch_curve = calibrate_feature_curve(features.pitch, low_q=0.10, high_q=0.95, gamma=1.0)
    subject_drive_curve = build_subject_drive_curve(
        energy_curve,
        bright_curve,
        pitch_curve,
        smoothing_window=subject_smoothing_window,
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
        canny_low=canny_low,
        canny_high=canny_high,
    )
    args_slug = _build_args_slug(run_args)

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
    print(f"Output: {output_dir}/")
    print(f"Resume state: {resume_state_file.name}")

    generator = ImageGenerator(
        ImageGenerationConfig(
            width=512,
            height=512,
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

    current = None
    start_frame = 0

    if resume_dir:
        loaded = _load_resume_state(output_dir, args_slug)
        if loaded is not None:
            loaded_run_args = loaded.get("run_args")
            if loaded_run_args and loaded_run_args != run_args:
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
            print(
                f"Resuming from frame {start_frame} using state "
                f"{Path(loaded['_state_file']).name}"
            )
        else:
            last_frame = _find_last_frame(output_dir)
            if last_frame >= 0:
                raise RuntimeError(
                    "Found frames in resume dir but no resume state file for current args. "
                    "Use the same args as before, or clear/change resume directory."
                )
            print("No resume state found. Starting from frame 0.")

    if start_frame == 0:
        if prompt_mode == "blend":
            subject_state = subject_controller.step(0, float(subject_drive_curve[0]))
            first_prompt = build_blended_prompt(
                prompt_drive=float(subject_drive_curve[0]),
                frame=0,
                total_frames=total_frames,
                beat_pulse=float(features.beat_pulse[0]),
                pitch=float(pitch_curve[0]),
                subject_blend=subject_state,
            )
        else:
            first_level = int(max(0, min(3, round(float(subject_drive_curve[0]) * 3.0))))
            first_prompt = build_discrete_prompt(
                prompt_level=first_level,
                frame=0,
                total_frames=total_frames,
                beat_pulse=float(features.beat_pulse[0]),
                pitch=float(pitch_curve[0]),
            )
        current = generator.generate_from_text(prompt=first_prompt, seed=42)
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
            cli_args=cli_args,
        )
        start_frame = 1

    if start_frame >= total_frames:
        print("All frames already generated; skipping frame generation.")

    for frame in range(start_frame, total_frames):
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

        transformed = transform_image(
            current,
            zoom=controls["zoom_delta"],
            angle=controls["angle_delta"],
            translation_x=controls["tx_delta"],
            translation_y=controls["ty_delta"],
        )

        if prompt_mode == "blend":
            subject_state = subject_controller.step(frame, float(subject_drive_curve[frame]))
            prompt = build_blended_prompt(
                prompt_drive=float(subject_drive_curve[frame]),
                frame=frame,
                total_frames=total_frames,
                beat_pulse=float(features.beat_pulse[frame]),
                pitch=float(pitch_curve[frame]),
                subject_blend=subject_state,
            )
            transition_active = bool(subject_state.in_transition)
        else:
            level = int(max(0, min(3, round(float(subject_drive_curve[frame]) * 3.0))))
            prompt = build_discrete_prompt(
                prompt_level=level,
                frame=frame,
                total_frames=total_frames,
                beat_pulse=float(features.beat_pulse[frame]),
                pitch=float(pitch_curve[frame]),
            )
            transition_active = False

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
        used_seed = 42 + frame * 97 + int(controls["seed_jump"])

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

            control_image = None
            if use_controlnet:
                control_image = build_canny_control_image(
                    transformed,
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                )

            noised = add_gaussian_noise(transformed, amount=used_noise, seed=used_seed + 17)
            current = generator.generate_from_image(
                init_image=noised,
                prompt=prompt,
                strength=used_strength,
                guidance_scale=used_cfg,
                seed=used_seed,
                control_image=control_image,
                controlnet_conditioning_scale=controls["control_scale"],
            )
        else:
            current = coherence_helper.apply(prev_frame=current, transformed_frame=transformed)

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
            cli_args=cli_args,
        )

        if (frame + 1) % 20 == 0 or frame == total_frames - 1:
            beat_tag = "BEAT" if frame in beat_set else "-"
            print(
                f"{frame:>4}/{total_frames-1} [{beat_tag}] "
                f"eng={energy_curve[frame]:.2f} onset={onset_curve[frame]:.2f} pitch={pitch_curve[frame]:.2f} "
                f"str={used_strength:.3f} cfg={used_cfg:.2f} "
                f"zoom={controls['zoom_delta']:.4f} pan={controls['tx_delta']:+.2f} "
                f"noise={used_noise:.3f} trans={int(transition_active)} "
                f"cscale={controls['control_scale']:.3f} cn={'ON' if use_controlnet else 'OFF'}"
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
    parser.add_argument("--prompt-mode", choices=["blend", "hard"], default="blend")
    parser.add_argument("--coherence", choices=["none", "blend", "optical_flow"], default="blend")
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
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
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
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        resume_dir=args.resume_dir,
        cli_args=sys.argv[1:],
    )
