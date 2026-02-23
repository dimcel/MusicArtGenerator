"""
Real Music Feature -> Video Test (Simple)

Uses a real audio file:
1) Extract audio features (beat/energy/onset/brightness)
2) Map features to controls (strength/cfg/zoom/pan/angle/prompt)
3) Run Deforum-like loop: transform -> img2img
4) Export video and mux original audio

Run:
  python test_real_music_feature_video.py --audio "/path/to/song.mp3" --with-audio
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "_src"))

from audio_feature_extractor import AudioFeatureExtractor
from cadence_coherence import CadenceCoherence
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
):
    extractor = AudioFeatureExtractor(audio_path=audio_path, fps=fps).load()

    if max_seconds is not None and max_seconds > 0:
        total_frames = int(min(extractor.duration_seconds, max_seconds) * fps)
    else:
        total_frames = int(extractor.duration_seconds * fps)
    total_frames = max(2, total_frames)

    features = extractor.extract(total_frames=total_frames)
    beat_set = set(features.beat_frames)
    mapper_cfg = MusicMappingConfig()

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

    safe_name = Path(audio_path).stem.replace(" ", "_")
    output_dir = Path(f"real_music_feature_video_{safe_name}_{mode}_{total_frames}f")
    output_dir.mkdir(exist_ok=True)

    print(f"Audio: {audio_path}")
    print(f"Duration used: {total_frames / fps:.2f}s ({total_frames} frames @ {fps}fps)")
    print(f"BPM: {features.bpm:.1f}, beats: {len(features.beat_frames)}")
    print(f"Mode: {mode}, cadence: {cadence}, prompt_mode: {prompt_mode}, coherence: {coherence}")
    print(f"Output: {output_dir}/")

    generator = ImageGenerator(ImageGenerationConfig(width=512, height=512))
    coherence_helper = CadenceCoherence(method=coherence, blend_alpha=0.35)
    subject_controller = SubjectTransitionController(
        num_subjects=4,
        hold_frames=subject_hold_frames,
        transition_frames=subject_transition_frames,
    )
    seed_state = 42.0

    first_controls = map_frame_to_controls(
        frame=0,
        total_frames=total_frames,
        energy=float(energy_curve[0]),
        onset=float(onset_curve[0]),
        brightness=float(bright_curve[0]),
        pitch=float(pitch_curve[0]),
        beat_pulse=float(features.beat_pulse[0]),
        cfg=mapper_cfg,
    )
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

    for frame in range(1, total_frames):
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
        if do_diffuse:
            target_seed = float(42 + frame * 97 + int(controls["seed_jump"]))
            # Seed travel-style interpolation during subject transitions.
            if transition_active:
                seed_state = seed_state + 0.25 * (target_seed - seed_state)
            else:
                seed_state = target_seed
            seed = int(seed_state)

            strength = float(controls["strength"])
            cfg_scale = float(controls["cfg_scale"])
            noise_amount = float(controls["noise_amount"])

            # Keep transitions stable: moderate strength, avoid spikes.
            if transition_active:
                strength = max(0.55, min(0.70, strength))
                cfg_scale = min(cfg_scale, 9.8)
                noise_amount = min(noise_amount, 0.05)

            noised = add_gaussian_noise(transformed, amount=noise_amount, seed=seed + 17)
            current = generator.generate_from_image(
                init_image=noised,
                prompt=prompt,
                strength=strength,
                guidance_scale=cfg_scale,
                seed=seed,
            )
        else:
            current = coherence_helper.apply(prev_frame=current, transformed_frame=transformed)

        current.save(output_dir / f"frame_{frame:05d}.png")

        if (frame + 1) % 20 == 0 or frame == total_frames - 1:
            beat_tag = "BEAT" if frame in beat_set else "-"
            print(
                f"{frame:>4}/{total_frames-1} [{beat_tag}] "
                f"eng={energy_curve[frame]:.2f} onset={onset_curve[frame]:.2f} pitch={pitch_curve[frame]:.2f} "
                f"str={controls['strength']:.3f} cfg={controls['cfg_scale']:.2f} "
                f"zoom={controls['zoom_delta']:.4f} pan={controls['tx_delta']:+.2f} "
                f"noise={controls['noise_amount']:.3f} trans={int(transition_active)}"
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
    )
