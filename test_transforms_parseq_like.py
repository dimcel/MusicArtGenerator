"""
Parseq-Like Transform Test

Similar to test_transforms.py (interpolation mode), but uses a small
parseq-like scheduler to drive per-frame parameters.

Run:
    python test_transforms_parseq_like.py
"""

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


def build_scheduler(total_frames, beat_frames):
    """
    Build a simple parseq-like schedule for transform/generation params.

    Notes:
    - keyframe curves define slow global trends
    - beat_pulse adds local modulation near beats
    """
    specs = {
        "zoom_delta": ParameterSpec(
            keyframes=[(0, 1.006), (total_frames // 2, 1.012), (total_frames - 1, 1.007)],
            easing="ease_in_out",
            beat_pulse_amount=0.008,
            beat_pulse_decay_frames=2,
            clamp=(1.001, 1.03),
        ),
        "pan_x_delta": ParameterSpec(
            expression="0.25*sin(2*pi*t*1.7) + 0.45*(beat_pulse - 0.3)",
            clamp=(-0.6, 0.8),
        ),
        "angle_delta": ParameterSpec(
            expression="-0.05 + 0.3*beat_pulse",
            clamp=(-0.2, 0.5),
        ),
        "strength": ParameterSpec(
            keyframes=[(0, 0.55), (total_frames // 2, 0.7), (total_frames - 1, 0.6)],
            easing="ease_in_out",
            beat_pulse_amount=0.16,
            beat_pulse_decay_frames=2,
            clamp=(0.35, 0.92),
        ),
        "cfg_scale": ParameterSpec(
            expression="7.2 + 1.3*sin(2*pi*t) + 1.8*beat_pulse",
            clamp=(6.0, 11.0),
        ),
        # Used for prompt switching at keyframes.
        "dramatic_score": ParameterSpec(
            expression="0.2 + 1.1*beat_pulse",
            clamp=(0.0, 1.3),
        ),
    }
    return ParseqLikeScheduler(specs=specs, beat_frames=beat_frames)


def generate_transform_video_parseq_like():
    print("\n" + "=" * 70)
    print("GENERATING PARSEQ-LIKE TRANSFORM VIDEO")
    print("=" * 70)

    prompt_small = "old man sitting on bench, peaceful autumn park, afternoon light"
    prompt_big = "old man sitting on bench, dramatic autumn park, golden hour, vibrant colors, sunny"

    total_frames = 120
    fps = 24
    bpm = 120
    output_dir = Path("transform_video_parseq_like")

    beat_frames, beat_intensities = generate_fake_beats(total_frames, fps=fps, bpm=bpm)
    scheduler = build_scheduler(total_frames=total_frames, beat_frames=beat_frames)

    print("\nConfiguration:")
    print(f"  Frames: {total_frames} ({total_frames / fps:.1f}s)")
    print(f"  FPS: {fps}")
    print(f"  Beat frames: {beat_frames}")
    print(f"  Beat sizes: {['SMALL' if x < 0.5 else 'BIG' for x in beat_intensities]}")
    print(f"  Output: {output_dir}/")

    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    interpolator = FrameInterpolator(method="blend")
    output_dir.mkdir(exist_ok=True)

    keyframe_images = {}

    # Frame 0
    current_image = generator.generate_from_text(prompt=prompt_small, seed=42)
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
            )

        beat_params = scheduler.frame_params(curr_beat, total_frames)
        prompt = prompt_big if beat_params["dramatic_score"] >= 0.7 else prompt_small
        prompt_label = "BIG" if prompt == prompt_big else "SMALL"

        current_image = generator.generate_from_image(
            init_image=current_image,
            prompt=prompt,
            strength=beat_params["strength"],
            guidance_scale=beat_params["cfg_scale"],
            seed=42 + i * 137,
        )

        keyframe_images[curr_beat] = current_image
        print(
            f"  Keyframe {curr_beat:>3}: prompt={prompt_label}, "
            f"zoom={beat_params['zoom_delta']:.4f}, "
            f"pan={beat_params['pan_x_delta']:+.3f}, "
            f"rot={beat_params['angle_delta']:+.3f}, "
            f"str={beat_params['strength']:.3f}, cfg={beat_params['cfg_scale']:.2f}"
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

        video_path = output_dir / "transform_parseq_like.mp4"
        frames_to_video(
            frames_dir=str(output_dir),
            output_path=str(video_path),
            fps=fps,
        )
        print(f"\nVideo created: {video_path}")
    except ImportError:
        print("\nCreate video with ffmpeg:")
        print(f"  cd {output_dir}")
        print(
            f"  ffmpeg -framerate {fps} -i frame_%05d.png "
            f"-c:v libx264 -pix_fmt yuv420p transform_parseq_like.mp4"
        )

    print("\nDone.")


if __name__ == "__main__":
    generate_transform_video_parseq_like()

