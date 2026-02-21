"""
Deforum-Like Music Loop (Simple)

Core loop per frame:
1) Transform previous frame (zoom/pan/rotate)
2) Run img2img from transformed frame
3) Save frame

This is closer to "moving through" the scene than a post-process zoom pass.

Run examples:
  python test_deforum_like_music_loop.py
  python test_deforum_like_music_loop.py --frames 240 --mode full --with-audio
  python test_deforum_like_music_loop.py --mode cadence --cadence 3 --with-audio
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "_src"))

from image_generator import ImageGenerationConfig, ImageGenerator
from image_transform import transform_image
from parseq_like_scheduler import ParameterSpec, ParseqLikeScheduler


def generate_fake_beats(total_frames, fps=24, bpm=124):
    import random

    random.seed(42)
    frames_per_beat = (60.0 / bpm) * fps
    beat_frames = [0]
    beat_intensities = [0.6]

    current = 0
    beat_count = 0
    while current < total_frames:
        jitter = frames_per_beat * random.uniform(-0.12, 0.12)
        current += int(frames_per_beat + jitter)
        if current < total_frames:
            beat_frames.append(current)
            beat_count += 1
            beat_intensities.append(0.35 if beat_count % 2 else 1.0)

    return beat_frames, beat_intensities


def choose_prompt(frame_num, total_frames, beat_intensity):
    phase = min(3, int((4 * frame_num) / max(1, total_frames)))
    big = beat_intensity >= 0.5

    small_prompts = [
        "elderly violinist near a park bench, cinematic realism, soft light",
        "young street dancer in urban plaza at dusk, film still, subtle neon haze",
        "female astronaut walking on alien shoreline, bioluminescent mist, cinematic",
        "samurai in dawn field, fog and warm rim light, high detail",
    ]
    big_prompts = [
        "elderly violinist in dramatic golden rays, swirling leaves, high contrast cinematic",
        "young street dancer mid-jump under neon signs, dynamic action frame, cinematic",
        "female astronaut running through alien storm, lightning sky, epic cinematic realism",
        "samurai charging through dawn fog, dust trails and god rays, sharp cinematic detail",
    ]
    return big_prompts[phase] if big else small_prompts[phase]


def build_scheduler(total_frames, beat_frames):
    specs = {
        # Deltas applied every frame before img2img.
        "zoom_delta": ParameterSpec(
            expression="1.004 + 0.010*beat_pulse + 0.004*sin(2*pi*t*2.0)",
            clamp=(0.995, 1.04),
        ),
        "angle_delta": ParameterSpec(
            expression="-0.10 + 1.8*sin(2*pi*t*1.5) + 2.8*beat_pulse",
            clamp=(-6.0, 6.0),
        ),
        "tx_delta": ParameterSpec(
            expression="1.6*sin(2*pi*t*2.2) + 2.4*(beat_pulse-0.2)",
            clamp=(-7.0, 7.0),
        ),
        "ty_delta": ParameterSpec(
            expression="0.9*cos(2*pi*t*1.4) + 1.6*beat_pulse",
            clamp=(-5.5, 5.5),
        ),
        # Diffusion controls.
        "strength": ParameterSpec(
            keyframes=[(0, 0.58), (total_frames // 2, 0.82), (total_frames - 1, 0.70)],
            easing="ease_in_out",
            beat_pulse_amount=0.14,
            beat_pulse_decay_frames=2,
            clamp=(0.45, 0.96),
        ),
        "cfg_scale": ParameterSpec(
            expression="7.2 + 1.5*sin(2*pi*t) + 2.2*beat_pulse",
            clamp=(6.0, 12.8),
        ),
    }
    return ParseqLikeScheduler(specs=specs, beat_frames=beat_frames)


def run(frames=120, fps=24, bpm=124, mode="full", cadence=3, with_audio=False):
    output_dir = Path(f"deforum_like_music_{mode}_{frames}f")
    output_dir.mkdir(exist_ok=True)

    beat_frames, beat_intensities = generate_fake_beats(frames, fps=fps, bpm=bpm)
    beat_map = {f: beat_intensities[i] for i, f in enumerate(beat_frames)}
    beat_set = set(beat_frames)
    scheduler = build_scheduler(frames, beat_frames)

    print(f"Frames={frames}, FPS={fps}, BPM={bpm}, Mode={mode}, Cadence={cadence}")
    print(f"Beats={len(beat_frames)}, Output={output_dir}/")

    generator = ImageGenerator(ImageGenerationConfig(width=512, height=512))

    first_prompt = choose_prompt(0, frames, beat_map.get(0, 0.5))
    current = generator.generate_from_text(prompt=first_prompt, seed=42)
    current.save(output_dir / "frame_00000.png")
    print("Frame 0 generated")

    for frame in range(1, frames):
        p = scheduler.frame_params(frame, frames)

        transformed = transform_image(
            current,
            zoom=p["zoom_delta"],
            angle=p["angle_delta"],
            translation_x=p["tx_delta"],
            translation_y=p["ty_delta"],
        )

        beat_intensity = beat_map.get(frame, 0.0)
        prompt = choose_prompt(frame, frames, beat_intensity)

        strength = min(0.98, p["strength"] + 0.10 * beat_intensity)
        cfg_scale = min(13.2, p["cfg_scale"] + 1.5 * beat_intensity)

        do_diffuse = (mode == "full") or (frame in beat_set) or (frame % cadence == 0)
        if do_diffuse:
            current = generator.generate_from_image(
                init_image=transformed,
                prompt=prompt,
                strength=strength,
                guidance_scale=cfg_scale,
                seed=42 + frame * 131,
            )
        else:
            # Cadence skip: keep motion evolving even if we skip img2img this frame.
            current = transformed

        current.save(output_dir / f"frame_{frame:05d}.png")
        if (frame + 1) % 20 == 0 or frame == frames - 1:
            beat_tag = "BEAT" if frame in beat_set else "-"
            print(
                f"{frame:>3}/{frames-1} [{beat_tag}] "
                f"zoom={p['zoom_delta']:.4f} tx={p['tx_delta']:+.2f} "
                f"rot={p['angle_delta']:+.2f} str={strength:.3f} cfg={cfg_scale:.2f}"
            )

    from frames_to_video import frames_to_video

    video_path = output_dir / f"deforum_like_music_{mode}_{frames}f.mp4"
    frames_to_video(str(output_dir), str(video_path), fps=fps)
    print(f"Video: {video_path}")

    if with_audio:
        from fake_noise_audio import generate_fake_music_track

        wav_path = output_dir / "fake_music.wav"
        audio_profile = "extended" if frames >= 200 else "base"
        generate_fake_music_track(
            output_path=str(wav_path),
            beat_frames=beat_frames,
            beat_intensities=beat_intensities,
            fps=fps,
            total_frames=frames,
            profile=audio_profile,
        )

        muxed_path = output_dir / f"deforum_like_music_{mode}_{frames}f_with_audio.mp4"
        cmd = (
            f'ffmpeg -y -i "{video_path}" -i "{wav_path}" '
            f'-c:v copy -c:a aac -shortest "{muxed_path}"'
        )
        if os.system(cmd) == 0:
            print(f"Muxed: {muxed_path}")
        else:
            print("ffmpeg mux failed; video and wav still exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--mode", choices=["full", "cadence"], default="full")
    parser.add_argument("--cadence", type=int, default=3)
    parser.add_argument("--with-audio", action="store_true")
    args = parser.parse_args()

    run(
        frames=args.frames,
        fps=args.fps,
        bpm=args.bpm,
        mode=args.mode,
        cadence=max(1, args.cadence),
        with_audio=args.with_audio,
    )

