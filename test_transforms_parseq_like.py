"""
Parseq-Like Transform Test

Similar to test_transforms.py (interpolation mode), but uses a small
parseq-like scheduler to drive per-frame parameters.

Run:
    python test_transforms_parseq_like.py
    python test_transforms_parseq_like.py --preset strong --with-audio
    python test_transforms_parseq_like.py --profile extended --preset strong --with-audio
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _src.image_generator import ImageGenerator, ImageGenerationConfig
from _src.image_transform import transform_image
from _src.frame_interpolator import FrameInterpolator
from _src.parseq_like_scheduler import ParseqLikeScheduler, ParameterSpec


def generate_fake_beats(total_frames, fps=24, bpm=120):
    """Generate beat frame positions and alternating intensities."""
    import random

    random.seed(42)
    frames_per_beat = (60.0 / bpm) * fps

    beat_frames = [0]
    beat_intensities = [0.5]

    current_frame = 0
    beat_count = 0
    while current_frame < total_frames:
        variation = frames_per_beat * random.uniform(-0.1, 0.1)
        current_frame += int(frames_per_beat + variation)
        if current_frame < total_frames:
            beat_frames.append(current_frame)
            beat_count += 1
            beat_intensities.append(0.3 if beat_count % 2 == 1 else 1.0)

    return beat_frames, beat_intensities


def build_scheduler(total_frames, beat_frames, preset="strong"):
    """
    Build a simple parseq-like schedule for transform/generation params.

    Notes:
    - keyframe curves define slow global trends
    - beat_pulse adds local modulation near beats
    """
    if preset == "subtle":
        specs = {
            "zoom_delta": ParameterSpec(
                keyframes=[(0, 1.005), (total_frames // 2, 1.010), (total_frames - 1, 1.006)],
                easing="ease_in_out",
                beat_pulse_amount=0.006,
                beat_pulse_decay_frames=2,
                clamp=(1.001, 1.02),
            ),
            "pan_x_delta": ParameterSpec(
                expression="0.22*sin(2*pi*t*1.4) + 0.35*(beat_pulse - 0.3)",
                clamp=(-0.6, 0.7),
            ),
            "angle_delta": ParameterSpec(
                expression="-0.04 + 0.20*beat_pulse",
                clamp=(-0.2, 0.35),
            ),
            "strength": ParameterSpec(
                keyframes=[(0, 0.52), (total_frames // 2, 0.66), (total_frames - 1, 0.58)],
                easing="ease_in_out",
                beat_pulse_amount=0.12,
                beat_pulse_decay_frames=2,
                clamp=(0.35, 0.9),
            ),
            "cfg_scale": ParameterSpec(
                expression="7.0 + 1.0*sin(2*pi*t) + 1.2*beat_pulse",
                clamp=(6.0, 10.5),
            ),
        }
    else:
        specs = {
            "zoom_delta": ParameterSpec(
                keyframes=[(0, 1.010), (total_frames // 2, 1.024), (total_frames - 1, 1.012)],
                easing="ease_in_out",
                beat_pulse_amount=0.012,
                beat_pulse_decay_frames=2,
                clamp=(1.003, 1.04),
            ),
            "pan_x_delta": ParameterSpec(
                expression="0.55*sin(2*pi*t*1.9) + 0.80*(beat_pulse - 0.25)",
                clamp=(-1.4, 1.5),
            ),
            "angle_delta": ParameterSpec(
                expression="-0.08 + 0.55*beat_pulse",
                clamp=(-0.35, 0.9),
            ),
            "strength": ParameterSpec(
                keyframes=[(0, 0.62), (total_frames // 2, 0.82), (total_frames - 1, 0.70)],
                easing="ease_in_out",
                beat_pulse_amount=0.20,
                beat_pulse_decay_frames=2,
                clamp=(0.45, 0.96),
            ),
            "cfg_scale": ParameterSpec(
                expression="7.4 + 1.8*sin(2*pi*t) + 2.4*beat_pulse",
                clamp=(6.2, 12.5),
            ),
        }
    return ParseqLikeScheduler(specs=specs, beat_frames=beat_frames)


def choose_prompt(frame_num, total_frames, beat_intensity):
    """
    Creative scene schedule:
    - 4 time-based phases
    - small/big beat prompt variants inside each phase
    """
    phase = min(3, int((4 * frame_num) / max(1, total_frames)))
    is_big = beat_intensity >= 0.5

    small_prompts = [
        "old man sitting on bench, quiet autumn park, soft afternoon light, cinematic",
        "old man on bench near city trees, blue dusk haze, moody atmosphere, film still",
        "old man on bench in light rain, reflective puddles, neon reflections, cinematic realism",
        "old man on bench at dawn, misty golden fog, hopeful mood, high detail",
    ]
    big_prompts = [
        "old man sitting on bench, dramatic golden hour burst, wind in leaves, vibrant cinematic grade",
        "old man on bench in city park, dramatic blue-orange contrast, headlights streaks, intense mood",
        "old man on bench in stormy rain, lightning glow, wet pavement shine, dramatic cinematic realism",
        "old man on bench at sunrise explosion, volumetric god rays, epic color contrast, sharp detail",
    ]
    return big_prompts[phase] if is_big else small_prompts[phase]


def generate_transform_video_parseq_like(preset="strong", with_audio=False, profile="base"):
    print("\n" + "=" * 70)
    print("GENERATING PARSEQ-LIKE TRANSFORM VIDEO")
    print("=" * 70)

    total_frames = 120 if profile == "base" else 240
    fps = 24
    bpm = 120 if profile == "base" else 126
    output_dir = Path(f"transform_video_parseq_like_{preset}_{profile}")

    beat_frames, beat_intensities = generate_fake_beats(total_frames, fps=fps, bpm=bpm)
    scheduler = build_scheduler(total_frames=total_frames, beat_frames=beat_frames, preset=preset)

    print("\nConfiguration:")
    print(f"  Frames: {total_frames} ({total_frames / fps:.1f}s)")
    print(f"  FPS: {fps}")
    print(f"  Preset: {preset}")
    print(f"  Profile: {profile}")
    print(f"  Beat frames: {beat_frames}")
    print(f"  Beat sizes: {['SMALL' if x < 0.5 else 'BIG' for x in beat_intensities]}")
    print(f"  Output: {output_dir}/")

    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    interpolator = FrameInterpolator(method="optical_flow")
    output_dir.mkdir(exist_ok=True)

    keyframe_images = {}

    # Frame 0
    first_prompt = choose_prompt(0, total_frames, beat_intensities[0])
    current_image = generator.generate_from_text(prompt=first_prompt, seed=42)
    keyframe_images[0] = current_image
    print("\nGenerated initial frame 0")

    # Generate beat keyframes using scheduled params
    for i in range(1, len(beat_frames)):
        prev_beat = beat_frames[i - 1]
        curr_beat = beat_frames[i]

        # Apply per-frame transforms from scheduler between beat keyframes
        for frame in range(prev_beat + 1, curr_beat + 1):
            params = scheduler.frame_params(frame, total_frames)
            current_image = transform_image(
                current_image,
                zoom=params["zoom_delta"],
                angle=params["angle_delta"],
                translation_x=params["pan_x_delta"],
                translation_y=0.35 * params["pan_x_delta"],
            )

        beat_params = scheduler.frame_params(curr_beat, total_frames)
        beat_intensity = beat_intensities[i]
        prompt = choose_prompt(curr_beat, total_frames, beat_intensity)
        prompt_label = "BIG" if beat_intensity >= 0.5 else "SMALL"
        strength = min(0.97, beat_params["strength"] + 0.15 * beat_intensity)
        cfg_scale = min(13.0, beat_params["cfg_scale"] + 1.8 * beat_intensity)

        current_image = generator.generate_from_image(
            init_image=current_image,
            prompt=prompt,
            strength=strength,
            guidance_scale=cfg_scale,
            seed=42 + i * 137,
        )

        keyframe_images[curr_beat] = current_image
        print(
            f"  Keyframe {curr_beat:>3}: prompt={prompt_label}, "
            f"zoom={beat_params['zoom_delta']:.4f}, "
            f"pan={beat_params['pan_x_delta']:+.3f}, "
            f"rot={beat_params['angle_delta']:+.3f}, "
            f"str={strength:.3f}, cfg={cfg_scale:.2f}, "
            f"txt='{prompt[:42]}...'"
        )

    # Interpolate between keyframes
    all_frames = []
    for i in range(len(beat_frames) - 1):
        frame_a_num = beat_frames[i]
        frame_b_num = beat_frames[i + 1]
        frame_a = keyframe_images[frame_a_num]
        frame_b = keyframe_images[frame_b_num]

        all_frames.append(frame_a)
        num_between = frame_b_num - frame_a_num - 1
        if num_between > 0:
            all_frames.extend(interpolator.interpolate(frame_a, frame_b, num_between))

    all_frames.append(keyframe_images[beat_frames[-1]])

    # Save frames
    for i, frame in enumerate(all_frames):
        frame.save(output_dir / f"frame_{i:05d}.png")
        if (i + 1) % 20 == 0 or i == len(all_frames) - 1:
            print(f"  Saved {i + 1}/{len(all_frames)} frames")

    # Create video
    try:
        from frames_to_video import frames_to_video

        video_path = output_dir / f"transform_parseq_like_{preset}_{profile}.mp4"
        frames_to_video(
            frames_dir=str(output_dir),
            output_path=str(video_path),
            fps=fps,
        )
        print(f"\nVideo created: {video_path}")

        if with_audio:
            from fake_noise_audio import generate_fake_music_track

            audio_path = output_dir / "fake_music.wav"
            generate_fake_music_track(
                output_path=str(audio_path),
                beat_frames=beat_frames,
                beat_intensities=beat_intensities,
                fps=fps,
                total_frames=total_frames,
            )

            muxed_path = output_dir / f"transform_parseq_like_{preset}_{profile}_with_audio.mp4"
            cmd = (
                f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
                f'-c:v copy -c:a aac -shortest "{muxed_path}"'
            )
            result = os.system(cmd)
            if result == 0:
                print(f"Audio muxed video: {muxed_path}")
            else:
                print("Could not mux audio with ffmpeg. Video and wav are still saved.")
    except ImportError:
        print("\nCreate video with ffmpeg:")
        print(f"  cd {output_dir}")
        print(
            f"  ffmpeg -framerate {fps} -i frame_%05d.png "
            f"-c:v libx264 -pix_fmt yuv420p transform_parseq_like_{preset}_{profile}.mp4"
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["subtle", "strong"], default="strong")
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    parser.add_argument("--with-audio", action="store_true")
    args = parser.parse_args()
    generate_transform_video_parseq_like(
        preset=args.preset,
        with_audio=args.with_audio,
        profile=args.profile,
    )
