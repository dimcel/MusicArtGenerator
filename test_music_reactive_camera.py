"""
Music-Reactive Camera Test

Goal:
- Keep diffusion keyframe generation simple.
- Add a second camera pass (zoom/rotate/pan) on EVERY final frame.
- Drive camera motion from parseq-like schedule + beat timing.

Run:
    python test_music_reactive_camera.py --profile extended --with-audio
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _src.frame_interpolator import FrameInterpolator
from _src.image_generator import ImageGenerationConfig, ImageGenerator
from _src.image_transform import transform_image
from _src.parseq_like_scheduler import ParameterSpec, ParseqLikeScheduler


def generate_fake_beats(total_frames, fps=24, bpm=124):
    import random

    random.seed(42)
    frames_per_beat = (60.0 / bpm) * fps

    beat_frames = [0]
    beat_intensities = [0.6]
    current_frame = 0
    beat_count = 0

    while current_frame < total_frames:
        variation = frames_per_beat * random.uniform(-0.12, 0.12)
        current_frame += int(frames_per_beat + variation)
        if current_frame < total_frames:
            beat_frames.append(current_frame)
            beat_count += 1
            beat_intensities.append(0.35 if beat_count % 2 else 1.0)

    return beat_frames, beat_intensities


def choose_prompt(frame_num, total_frames, beat_intensity):
    phase = min(3, int((4 * frame_num) / max(1, total_frames)))
    big = beat_intensity >= 0.5
    small = [
        "portrait of elderly violinist by bench, gentle afternoon park, cinematic realism",
        "young dancer in urban plaza at dusk, subtle neon haze, film still",
        "female astronaut walking through alien tide pools, calm bioluminescent mist, cinematic",
        "samurai in dawn grass field, soft fog and warm rim light, high detail",
    ]
    large = [
        "elderly violinist in dramatic wind and sun rays, flying leaves, high-contrast cinematic",
        "young dancer mid-jump with neon motion streaks, dramatic city energy, cinematic action frame",
        "female astronaut sprinting on alien shore during electric storm, epic lightning sky",
        "samurai charge through dawn fog with dust trails and volumetric rays, dramatic cinematic frame",
    ]
    return large[phase] if big else small[phase]


def build_generation_scheduler(total_frames, beat_frames):
    specs = {
        "strength": ParameterSpec(
            keyframes=[(0, 0.60), (total_frames // 2, 0.84), (total_frames - 1, 0.72)],
            easing="ease_in_out",
            beat_pulse_amount=0.16,
            beat_pulse_decay_frames=2,
            clamp=(0.45, 0.97),
        ),
        "cfg_scale": ParameterSpec(
            expression="7.3 + 1.4*sin(2*pi*t) + 2.2*beat_pulse",
            clamp=(6.0, 13.0),
        ),
    }
    return ParseqLikeScheduler(specs=specs, beat_frames=beat_frames)


def build_camera_scheduler(total_frames, beat_frames, intensity="strong"):
    if intensity == "subtle":
        specs = {
            "zoom": ParameterSpec(
                expression="1.002 + 0.004*beat_pulse + 0.002*sin(2*pi*t*2.0)",
                clamp=(0.995, 1.02),
            ),
            "angle": ParameterSpec(
                expression="0.7*sin(2*pi*t*1.2) + 0.8*beat_pulse",
                clamp=(-2.2, 2.2),
            ),
            "tx": ParameterSpec(
                expression="1.2*sin(2*pi*t*1.8) + 1.3*(beat_pulse-0.2)",
                clamp=(-5.0, 5.0),
            ),
            "ty": ParameterSpec(
                expression="0.8*cos(2*pi*t*1.1) + 0.8*beat_pulse",
                clamp=(-4.0, 4.0),
            ),
        }
    else:
        specs = {
            "zoom": ParameterSpec(
                expression="1.004 + 0.010*beat_pulse + 0.004*sin(2*pi*t*2.4)",
                clamp=(0.99, 1.05),
            ),
            "angle": ParameterSpec(
                expression="1.6*sin(2*pi*t*1.6) + 2.8*beat_pulse",
                clamp=(-6.0, 6.0),
            ),
            "tx": ParameterSpec(
                expression="2.8*sin(2*pi*t*2.2) + 3.2*(beat_pulse-0.25)",
                clamp=(-12.0, 12.0),
            ),
            "ty": ParameterSpec(
                expression="1.5*cos(2*pi*t*1.4) + 2.6*beat_pulse",
                clamp=(-8.0, 8.0),
            ),
        }
    return ParseqLikeScheduler(specs=specs, beat_frames=beat_frames)


def run(profile="extended", camera_intensity="strong", with_audio=False):
    total_frames = 240 if profile == "extended" else 120
    fps = 24
    bpm = 126 if profile == "extended" else 120
    output_dir = Path(f"music_reactive_camera_{profile}_{camera_intensity}")
    output_dir.mkdir(exist_ok=True)

    beat_frames, beat_intensities = generate_fake_beats(total_frames, fps=fps, bpm=bpm)
    gen_sched = build_generation_scheduler(total_frames, beat_frames)
    cam_sched = build_camera_scheduler(total_frames, beat_frames, intensity=camera_intensity)

    print(f"Frames={total_frames}, beats={len(beat_frames)}, profile={profile}, camera={camera_intensity}")

    generator = ImageGenerator(ImageGenerationConfig(width=512, height=512))
    interpolator = FrameInterpolator(method="optical_flow")

    # Keyframe generation at beat positions.
    keyframes = {}
    first_prompt = choose_prompt(0, total_frames, beat_intensities[0])
    current_image = generator.generate_from_text(prompt=first_prompt, seed=42)
    keyframes[0] = current_image

    for i in range(1, len(beat_frames)):
        frame_num = beat_frames[i]
        beat_intensity = beat_intensities[i]
        prompt = choose_prompt(frame_num, total_frames, beat_intensity)
        p = gen_sched.frame_params(frame_num, total_frames)
        strength = min(0.98, p["strength"] + 0.10 * beat_intensity)
        cfg = min(13.5, p["cfg_scale"] + 1.6 * beat_intensity)

        current_image = generator.generate_from_image(
            init_image=current_image,
            prompt=prompt,
            strength=strength,
            guidance_scale=cfg,
            seed=42 + i * 173,
        )
        keyframes[frame_num] = current_image
        print(f"Keyframe {frame_num:>3}: str={strength:.3f} cfg={cfg:.2f} beat={beat_intensity:.2f}")

    # Interpolate full frame list.
    all_frames = []
    for i in range(len(beat_frames) - 1):
        a_num = beat_frames[i]
        b_num = beat_frames[i + 1]
        all_frames.append(keyframes[a_num])
        between = b_num - a_num - 1
        if between > 0:
            all_frames.extend(interpolator.interpolate(keyframes[a_num], keyframes[b_num], between))
    all_frames.append(keyframes[beat_frames[-1]])

    # Camera pass on EVERY final frame.
    camera_frames = []
    for i, frame in enumerate(all_frames):
        c = cam_sched.frame_params(i, total_frames)
        cam_frame = transform_image(
            frame,
            zoom=c["zoom"],
            angle=c["angle"],
            translation_x=c["tx"],
            translation_y=c["ty"],
        )
        camera_frames.append(cam_frame)

    # Save frames.
    for i, frame in enumerate(camera_frames):
        frame.save(output_dir / f"frame_{i:05d}.png")
        if (i + 1) % 30 == 0 or i == len(camera_frames) - 1:
            print(f"Saved {i + 1}/{len(camera_frames)}")

    # Video.
    from frames_to_video import frames_to_video

    video_path = output_dir / f"music_reactive_camera_{profile}_{camera_intensity}.mp4"
    frames_to_video(str(output_dir), str(video_path), fps=fps)
    print(f"Video: {video_path}")

    if with_audio:
        from fake_noise_audio import generate_fake_music_track

        wav_path = output_dir / "fake_music.wav"
        generate_fake_music_track(
            output_path=str(wav_path),
            beat_frames=beat_frames,
            beat_intensities=beat_intensities,
            fps=fps,
            total_frames=total_frames,
            profile=profile,
        )
        muxed = output_dir / f"music_reactive_camera_{profile}_{camera_intensity}_with_audio.mp4"
        cmd = (
            f'ffmpeg -y -i "{video_path}" -i "{wav_path}" '
            f'-c:v copy -c:a aac -shortest "{muxed}"'
        )
        if os.system(cmd) == 0:
            print(f"Muxed: {muxed}")
        else:
            print("ffmpeg mux failed; video and wav still exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["base", "extended"], default="extended")
    parser.add_argument("--camera-intensity", choices=["subtle", "strong"], default="strong")
    parser.add_argument("--with-audio", action="store_true")
    args = parser.parse_args()
    run(profile=args.profile, camera_intensity=args.camera_intensity, with_audio=args.with_audio)

