"""Main generation runtime for the music-reactive video app."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from music_art_generator.audio_feature_extractor import AudioFeatureExtractor
from music_art_generator.cadence_coherence import CadenceCoherence
from music_art_generator.color_coherence import apply_color_coherence
from music_art_generator.controlnet_conditioning import build_canny_control_image
from music_art_generator.image_noise import add_gaussian_noise
from music_art_generator.image_transform import transform_image
from music_art_generator.music_feature_mapper import (
    MusicMappingConfig,
    calibrate_feature_curve,
    map_frame_to_controls,
)
from .music_logic import (
    _apply_music_color_fx,
    _apply_onset_jitter_controls,
    _build_onset_twist_gain,
    _build_quiet_hold_mask,
    _build_soft_run_gain,
    _load_and_resize_init_image,
    _parse_prompt_candidates,
    _re_anchor_profile,
)
from .resume import (
    _build_args_slug,
    _build_run_args,
    _changed_run_arg_keys,
    _find_last_frame,
    _load_resume_state,
    _save_resume_state,
    _state_path,
)

try:
    from music_art_generator.image_generator import ImageGenerationConfig, ImageGenerator
except ModuleNotFoundError:
    ImageGenerationConfig = None
    ImageGenerator = None


def _build_output_base_name(mode: str, total_frames: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_tag = f"{int(total_frames)}f"
    return f"output_{mode}_{timestamp}_{frames_tag}"


def _ensure_unique_dir_name(base_name: str) -> str:
    candidate = base_name
    suffix = 1
    while Path(candidate).exists():
        candidate = f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _ensure_unique_output_base_name(output_dir: Path, base_name: str) -> str:
    candidate = base_name
    suffix = 1
    while (
        (output_dir / f"{candidate}.mp4").exists()
        or (output_dir / f"{candidate}_with_audio.mp4").exists()
        or (output_dir / f"{candidate}.json").exists()
    ):
        candidate = f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _save_run_manifest(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def run(
    audio_path: str,
    fps: int = 24,
    device: str = "auto",
    mode: str = "full",
    cadence: int = 3,
    prompt_mode: str = "blend",
    coherence: str = "blend",
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
    steady_twist: bool = False,
    steady_twist_max_deg: float = 4.0,
    quiet_hold: bool = False,
    resume_dir: Optional[str] = None,
    cli_args: Optional[list] = None,
):
    extractor = AudioFeatureExtractor(audio_path=audio_path, fps=fps).load()

    audio_total_frames = max(2, int(extractor.duration_seconds * fps))
    if max_seconds is not None and max_seconds > 0:
        requested_frames = int(min(extractor.duration_seconds, max_seconds) * fps)
    else:
        requested_frames = audio_total_frames
    requested_frames = max(2, requested_frames)
    total_frames = requested_frames

    # Resume runs may extend the target after reading the saved state.
    # Keep full-track features available for safe indexing in that case.
    feature_frames = audio_total_frames if resume_dir else requested_frames

    features = extractor.extract(total_frames=feature_frames)
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
    # TODO(phase-2): `prompt_mode` is still carried for CLI/state compatibility.
    # It is not currently switching prompt-construction strategies in the frame loop.
    prompt_reactive_enabled = len(prompt_candidates) > 1 and (not disable_music_change)

    steady_twist_max_deg = abs(float(steady_twist_max_deg))
    quiet_hold_mask = None
    quiet_hold_gain = None
    quiet_hold_meta = None
    if quiet_hold:
        quiet_hold_mask, quiet_hold_meta = _build_quiet_hold_mask(
            energy_curve=energy_curve,
            onset_curve=onset_curve,
            fps=fps,
        )
        quiet_fade_frames = max(2, int(round(0.35 * fps)))
        quiet_hold_gain = _build_soft_run_gain(quiet_hold_mask, fade_frames=quiet_fade_frames)
        quiet_hold_meta["fade_frames"] = int(quiet_fade_frames)
    twist_onset_gain = None
    twist_onset_meta = None
    if steady_twist:
        twist_onset_gain, twist_onset_meta = _build_onset_twist_gain(
            onset_curve=onset_curve,
            fps=fps,
        )

    run_args = _build_run_args(
        audio_path=audio_path,
        fps=fps,
        device=device,
        mode=mode,
        cadence=cadence,
        prompt_mode=prompt_mode,
        coherence=coherence,
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
        steady_twist=steady_twist,
        steady_twist_max_deg=steady_twist_max_deg,
        quiet_hold=quiet_hold,
    )
    args_slug = _build_args_slug(run_args)
    width, height = 512, 512

    run_started_at = datetime.now().isoformat(timespec="seconds")
    if resume_dir:
        output_dir = Path(resume_dir)
    else:
        output_base_name = _ensure_unique_dir_name(
            _build_output_base_name(mode=mode, total_frames=total_frames)
        )
        output_dir = Path(output_base_name)
    output_dir.mkdir(exist_ok=True)
    resume_state_file = _state_path(output_dir, args_slug)

    print(f"Audio: {audio_path}")
    print(
        f"Duration requested: {requested_frames / fps:.2f}s "
        f"({requested_frames} frames @ {fps}fps)"
    )
    print(f"BPM: {features.bpm:.1f}, beats: {len(features.beat_frames)}")
    print(f"Device: {device}")
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
        print(f"Init image: {init_image}")
    else:
        print("Init image: OFF")
    print(f"Music color FX: {'ON' if music_color_fx else 'OFF'}")
    print(f"Onset jitter: {'ON' if onset_jitter else 'OFF'}")
    if quiet_hold and quiet_hold_meta is not None:
        print(
            "Quiet hold: ON "
            f"(eg<={quiet_hold_meta['energy_gate']:.2f}, "
            f"on<={quiet_hold_meta['onset_gate']:.2f}, "
            f"min_run={quiet_hold_meta['min_run_frames']}f, "
            f"fade={quiet_hold_meta['fade_frames']}f, "
            f"runs={quiet_hold_meta['quiet_runs']}, "
            f"active_f={quiet_hold_meta['quiet_frame_ratio']:.2f})"
        )
    else:
        print("Quiet hold: OFF")
    print(f"Color coherence: {color_coherence_mode} (strength={color_coherence_strength:.2f})")
    print(
        f"Re-anchor: {'ON' if re_anchor else 'OFF'} "
        f"(strength={re_anchor_strength}, every={max(1, int(re_anchor_every_frames))}f)"
    )
    if steady_twist:
        if twist_onset_meta is not None:
            print(
                "Steady twist: ON "
                f"(trigger=onset, thr={twist_onset_meta['threshold']:.2f}, "
                f"decay={twist_onset_meta['decay_frames']}f, "
                f"active_f={twist_onset_meta['active_frame_ratio']:.2f}, "
                f"max_deg={steady_twist_max_deg:.2f})"
            )
        else:
            print(f"Steady twist: ON (trigger=onset, max_deg={steady_twist_max_deg:.2f})")
    else:
        print("Steady twist: OFF")
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

    if Image is None:
        raise ImportError("Pillow is required to run the music video generator.")

    if ImageGenerator is None or ImageGenerationConfig is None:
        raise ImportError(
            "Image generation dependencies are missing. "
            "Install required runtime packages (e.g. torch/diffusers) to run generation."
        )

    generator = ImageGenerator(
        ImageGenerationConfig(
            width=width,
            height=height,
            device=device,
            enable_controlnet=use_controlnet,
            controlnet_type="canny",
            controlnet_model_id=controlnet_model,
        )
    )
    coherence_helper = CadenceCoherence(method=coherence, blend_alpha=0.35)
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
                loaded_run_args_cmp.pop("max_seconds", None)
            else:
                loaded_run_args_cmp = None

            if loaded_run_args_cmp:
                changed_keys = _changed_run_arg_keys(loaded_run_args_cmp, run_args)
                if changed_keys:
                    print(
                        "Resume note: run arguments differ from saved state; "
                        "continuing with new values."
                    )
                    print(f"Changed args ({len(changed_keys)}): {', '.join(changed_keys)}")
            saved_fps = loaded.get("fps")
            if saved_fps is not None and int(saved_fps) != int(fps):
                print(
                    "Resume note: state was created with "
                    f"{int(saved_fps)}fps but this run uses {int(fps)}fps."
                )
            last_frame = int(loaded.get("last_completed_frame", -1))
            frame_path = output_dir / f"frame_{last_frame:05d}.png"
            if not frame_path.exists():
                raise RuntimeError(
                    f"Resume state points to missing frame: {frame_path}"
                )
            if "_state_file_candidates" in loaded:
                candidate_names = loaded.get("_state_file_candidates", [])
                print(
                    "Resume note: multiple state files found; selected the most advanced one "
                    f"({Path(str(loaded['_state_file'])).name})."
                )
                if candidate_names:
                    print("Available states: " + ", ".join(candidate_names))
            # Keep writing updates to the exact state file we resumed from.
            resume_state_file = Path(str(loaded["_state_file"]))
            current = Image.open(frame_path).convert("RGB")
            start_frame = last_frame + 1

            if max_seconds is not None and max_seconds > 0:
                total_frames = min(audio_total_frames, start_frame + requested_frames)
                generated_now = max(0, total_frames - start_frame)
                print(
                    "Resume extension: "
                    f"requested +{max_seconds:.2f}s, "
                    f"scheduled +{generated_now / fps:.2f}s "
                    f"(target {total_frames / fps:.2f}s total)"
                )
            else:
                total_frames = audio_total_frames
                print(
                    "Resume extension: full remaining audio "
                    f"(target {total_frames / fps:.2f}s total)"
                )
            seed_state = float(loaded.get("seed_state", 42.0))
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
                    "Found frames in resume dir but no resume state file. "
                    "Clear/change resume directory or restore a resume_state_*.json file."
                )
            print("No resume state found. Starting from frame 0.")
    print(f"Generation target: {total_frames / fps:.2f}s ({total_frames} frames @ {fps}fps)")

    if resume_dir:
        output_base_name = _ensure_unique_output_base_name(
            output_dir,
            _build_output_base_name(mode=mode, total_frames=total_frames),
        )
    else:
        output_base_name = output_dir.name or _build_output_base_name(
            mode=mode, total_frames=total_frames
        )

    run_manifest_file = output_dir / f"{output_base_name}.json"
    video_path = output_dir / f"{output_base_name}.mp4"
    muxed_path = output_dir / f"{output_base_name}_with_audio.mp4"
    print(f"Export base: {output_base_name}")
    print(f"Run manifest: {run_manifest_file.name}")

    _save_run_manifest(
        run_manifest_file,
        {
            "schema_version": 1,
            "output_base_name": output_base_name,
            "output_dir": str(output_dir),
            "run_started_at": run_started_at,
            "last_updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_status": "running",
            "args_slug": args_slug,
            "audio_path": str(audio_path),
            "fps": int(fps),
            "mode": str(mode),
            "requested_frames": int(requested_frames),
            "target_total_frames": int(total_frames),
            "start_frame": int(start_frame),
            "resume_enabled": bool(resume_dir),
            "resume_state_file": str(resume_state_file),
            "run_args": run_args,
            "cli_args": cli_args or [],
            "video_path": str(video_path),
            "video_with_audio_path": str(muxed_path) if with_audio else None,
        },
    )

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

    steady_twist_deg = 0.0

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
        quiet_gain = 0.0
        if bool(quiet_hold) and (not disable_music_change) and (quiet_hold_gain is not None):
            if frame < int(quiet_hold_gain.size):
                quiet_gain = float(quiet_hold_gain[frame])
        quiet_active = (
            quiet_gain >= 0.55
        )
        if quiet_gain > 1e-6:
            # Soft quiet-hold: blend towards calmer controls.
            zoom_keep = 1.0 - 0.80 * quiet_gain
            pan_keep = 1.0 - 0.90 * quiet_gain
            angle_keep = 1.0 - 0.80 * quiet_gain
            noise_keep = 1.0 - 0.70 * quiet_gain

            controls["zoom_delta"] = float(1.0 + (float(controls["zoom_delta"]) - 1.0) * zoom_keep)
            controls["tx_delta"] = float(pan_keep * float(controls["tx_delta"]))
            controls["ty_delta"] = float(pan_keep * float(controls["ty_delta"]))
            controls["angle_delta"] = float(angle_keep * float(controls["angle_delta"]))
            controls["noise_amount"] = float(noise_keep * float(controls["noise_amount"]))
            controls["zoom_delta"] = max(
                float(mapper_cfg.zoom_min),
                min(float(mapper_cfg.zoom_max), float(controls["zoom_delta"])),
            )
            controls["tx_delta"] = max(
                -float(mapper_cfg.pan_abs_max),
                min(float(mapper_cfg.pan_abs_max), float(controls["tx_delta"])),
            )
            controls["ty_delta"] = max(
                -float(mapper_cfg.pan_abs_max),
                min(float(mapper_cfg.pan_abs_max), float(controls["ty_delta"])),
            )
            controls["angle_delta"] = max(
                -float(mapper_cfg.angle_abs_max),
                min(float(mapper_cfg.angle_abs_max), float(controls["angle_delta"])),
            )
            controls["noise_amount"] = max(
                float(mapper_cfg.noise_min),
                min(float(mapper_cfg.noise_max), float(controls["noise_amount"])),
            )

        twist_gain = 0.0
        if (
            steady_twist
            and (not disable_music_change)
            and (twist_onset_gain is not None)
            and frame < int(twist_onset_gain.size)
        ):
            twist_gain = float(twist_onset_gain[frame])
        target_twist = steady_twist_max_deg * twist_gain
        # Smooth right-twist pulse from onset spikes.
        steady_twist_deg = float(steady_twist_deg + 0.45 * (float(target_twist) - steady_twist_deg))
        if steady_twist and abs(steady_twist_deg) > 1e-6:
            controls["angle_delta"] = float(controls["angle_delta"] + steady_twist_deg)
            ang_cap = max(float(mapper_cfg.angle_abs_max), steady_twist_max_deg * 2.0)
            controls["angle_delta"] = max(-ang_cap, min(ang_cap, float(controls["angle_delta"])))

        onset_jitter_active = 0
        jitter_strength = 1.0 - quiet_gain
        if onset_jitter and not disable_music_change and jitter_strength > 1e-6:
            if float(onset_curve[frame]) >= 0.35 and jitter_strength > 0.05:
                onset_jitter_active = 1
            controls = _apply_onset_jitter_controls(
                controls=controls,
                onset_value=float(onset_curve[frame]),
                frame=frame,
                mapper_cfg=mapper_cfg,
                jitter_strength=jitter_strength,
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
        if prompt_reactive_enabled and (frame in beat_set) and (not quiet_active):
            beat_events_seen_for_prompt += 1
            if beat_events_seen_for_prompt % prompt_change_every_beats == 0:
                prev_prompt_idx = active_prompt_idx
                active_prompt_idx = (active_prompt_idx + 1) % len(prompt_candidates)
                prompt_switched = int(active_prompt_idx != prev_prompt_idx)

        prompt = prompt_candidates[active_prompt_idx]
        # Adaptive cadence:
        # diffuse in full mode, on cadence frames, and on beats.
        do_diffuse = (
            (mode == "full")
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
            seed_state = target_seed
            used_seed = int(seed_state)

            control_image = None
            if use_controlnet:
                control_image = build_canny_control_image(
                    transformed,
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                )

            noised = add_gaussian_noise(transformed, amount=used_noise, seed=used_seed + 17)

            # Periodically re-anchor to warped frame0 when init-image is provided.
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
                f"tw={steady_twist_deg:+.2f} twg={twist_gain:.2f} "
                f"qh={int(quiet_active)} qg={quiet_gain:.2f} jit={onset_jitter_active} "
                f"pidx={active_prompt_idx} psw={prompt_switched} "
                f"noise={used_noise:.3f} "
                f"cscale={used_control_scale:.3f} cn={'ON' if use_controlnet else 'OFF'} "
                f"ra={int(re_anchor_applied)}"
            )

    from music_art_generator.frames_to_video import frames_to_video

    frames_to_video(str(output_dir), str(video_path), fps=fps)
    print(f"Video: {video_path}")

    if with_audio:
        cmd = (
            f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
            f'-c:v copy -c:a aac -shortest "{muxed_path}"'
        )
        if os.system(cmd) == 0:
            print(f"Muxed: {muxed_path}")
        else:
            print("ffmpeg mux failed; silent video is still saved.")

    _save_run_manifest(
        run_manifest_file,
        {
            "schema_version": 1,
            "output_base_name": output_base_name,
            "output_dir": str(output_dir),
            "run_started_at": run_started_at,
            "last_updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_status": "completed",
            "args_slug": args_slug,
            "audio_path": str(audio_path),
            "fps": int(fps),
            "mode": str(mode),
            "requested_frames": int(requested_frames),
            "target_total_frames": int(total_frames),
            "resume_enabled": bool(resume_dir),
            "resume_state_file": str(resume_state_file),
            "run_args": run_args,
            "cli_args": cli_args or [],
            "video_path": str(video_path),
            "video_exists": bool(video_path.exists()),
            "video_with_audio_path": str(muxed_path) if with_audio else None,
            "video_with_audio_exists": bool(muxed_path.exists()) if with_audio else False,
        },
    )
