"""Main generation runtime for the music-reactive video app."""

import os
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from src.audio_feature_extractor import AudioFeatureExtractor
from src.cadence_coherence import CadenceCoherence
from src.color_coherence import apply_color_coherence
from src.controlnet_conditioning import build_canny_control_image
from src.image_noise import add_gaussian_noise
from src.image_transform import transform_image
from src.music_feature_mapper import (
    MusicMappingConfig,
    calibrate_feature_curve,
    map_frame_to_controls,
)
from src.prompt_blender import SubjectTransitionController

from .music_logic import (
    _apply_identity_anchor,
    _apply_music_color_fx,
    _apply_onset_jitter_controls,
    _build_auto_steady_activation_mask,
    _build_onset_twist_gain,
    _build_quiet_hold_mask,
    _build_soft_run_gain,
    _build_stability_curve,
    _build_stable_run_lengths,
    _load_and_resize_init_image,
    _parse_prompt_candidates,
    _re_anchor_profile,
    _steady_direction_from_features,
)
from .resume import (
    _build_args_slug,
    _build_run_args,
    _changed_run_arg_keys,
    _controller_from_dict,
    _find_last_frame,
    _load_resume_state,
    _save_resume_state,
    _state_path,
)

try:
    from src.image_generator import ImageGenerationConfig, ImageGenerator
except ModuleNotFoundError:
    ImageGenerationConfig = None
    ImageGenerator = None


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

    steady_activation_mode = str(steady_activation_mode or "auto").strip().lower()
    if steady_activation_mode not in ("auto", "manual"):
        raise ValueError("steady_activation_mode must be 'auto' or 'manual'")

    steady_activation_ratio = max(0.0, min(1.0, float(steady_activation_ratio)))
    steady_shift_probability = max(0.0, min(1.0, float(steady_shift_probability)))
    steady_threshold = max(0.0, min(1.0, float(steady_threshold)))
    steady_shift_pixels = abs(float(steady_shift_pixels))
    steady_twist_max_deg = abs(float(steady_twist_max_deg))
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
        steady_twist=steady_twist,
        steady_twist_max_deg=steady_twist_max_deg,
        quiet_hold=quiet_hold,
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
    print(
        f"Duration requested: {requested_frames / fps:.2f}s "
        f"({requested_frames} frames @ {fps}fps)"
    )
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
            enable_controlnet=use_controlnet,
            controlnet_type="canny",
            controlnet_model_id=controlnet_model,
        )
    )
    coherence_helper = CadenceCoherence(method=coherence, blend_alpha=0.35)
    # TODO(phase-2): subject transition controller state is persisted for backward
    # compatibility, but transition stepping is currently not wired into prompts.
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
                    "steady_twist",
                    "steady_twist_max_deg",
                    "steady_activation_mode",
                    "steady_activation_ratio",
                    "steady_shift_probability",
                    "music_color_fx",
                    "onset_jitter",
                    "steady_min_seconds",
                    "steady_threshold",
                    "steady_shift_pixels",
                    "quiet_hold",
                ):
                    if k not in loaded_run_args_cmp and k in run_args:
                        loaded_run_args_cmp[k] = run_args[k]
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
                    "Found frames in resume dir but no resume state file. "
                    "Clear/change resume directory or restore a resume_state_*.json file."
                )
            print("No resume state found. Starting from frame 0.")
    print(f"Generation target: {total_frames / fps:.2f}s ({total_frames} frames @ {fps}fps)")

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

        steady_active = False
        steady_direction = "none"
        if steady_shift and not disable_music_change and not quiet_active:
            if steady_activation_mode == "auto":
                steady_active = bool(steady_active_mask[frame]) if steady_active_mask is not None else False
            else:
                steady_active = (
                    bool(stable_run_lengths is not None)
                    and int(stable_run_lengths[frame]) >= steady_min_frames
                )

            if steady_active:
                stability_gain = float(stability_curve[frame])
                if not steady_prev_active:
                    steady_dir_label, steady_dir_x, steady_dir_y = _steady_direction_from_features(
                        pitch=float(pitch_curve[frame]),
                        brightness=float(bright_curve[frame]),
                    )
                steady_direction = steady_dir_label
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

        steady_prev_active = bool(steady_active)
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
        # TODO(phase-2): wire transition_active from subject_controller.step(...)
        # once prompt transition behavior is re-enabled.
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
                f"stab={stability_curve[frame]:.2f} sact={int(steady_active)} sdir={steady_direction} tw={steady_twist_deg:+.2f} twg={twist_gain:.2f} "
                f"qh={int(quiet_active)} qg={quiet_gain:.2f} jit={onset_jitter_active} "
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
