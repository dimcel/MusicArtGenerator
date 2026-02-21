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
from image_generator import ImageGenerationConfig, ImageGenerator
from image_transform import transform_image
from music_feature_mapper import MusicMappingConfig, map_frame_to_controls


def choose_prompt(prompt_level: int, frame: int, total_frames: int, beat_pulse: float) -> str:
    # Subject driven by audio level.
    subjects = [
        "elderly violinist",
        "young street dancer",
        "female astronaut",
        "samurai warrior",
    ]
    subject = subjects[max(0, min(3, int(prompt_level)))]

    # Scene driven by timeline phase.
    phase = min(3, int((4 * frame) / max(1, total_frames)))
    scenes = [
        "in an autumn park with cinematic soft light",
        "in a neon city plaza at blue dusk",
        "on an alien shoreline with bioluminescent mist",
        "in a dawn field with fog and volumetric rays",
    ]
    scene = scenes[phase]

    mood = "dramatic high-contrast action frame" if beat_pulse > 0.65 else "cinematic realism, detailed"
    return f"{subject}, {scene}, {mood}"


def run(
    audio_path: str,
    fps: int = 24,
    mode: str = "full",
    cadence: int = 3,
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

    safe_name = Path(audio_path).stem.replace(" ", "_")
    output_dir = Path(f"real_music_feature_video_{safe_name}_{mode}_{total_frames}f")
    output_dir.mkdir(exist_ok=True)

    print(f"Audio: {audio_path}")
    print(f"Duration used: {total_frames / fps:.2f}s ({total_frames} frames @ {fps}fps)")
    print(f"BPM: {features.bpm:.1f}, beats: {len(features.beat_frames)}")
    print(f"Mode: {mode}, cadence: {cadence}, output: {output_dir}/")

    generator = ImageGenerator(ImageGenerationConfig(width=512, height=512))

    first_controls = map_frame_to_controls(
        frame=0,
        total_frames=total_frames,
        energy=float(features.energy[0]),
        onset=float(features.onset[0]),
        brightness=float(features.brightness[0]),
        beat_pulse=float(features.beat_pulse[0]),
        cfg=mapper_cfg,
    )
    first_prompt = choose_prompt(
        prompt_level=int(first_controls["prompt_level"]),
        frame=0,
        total_frames=total_frames,
        beat_pulse=float(features.beat_pulse[0]),
    )
    current = generator.generate_from_text(prompt=first_prompt, seed=42)
    current.save(output_dir / "frame_00000.png")

    for frame in range(1, total_frames):
        controls = map_frame_to_controls(
            frame=frame,
            total_frames=total_frames,
            energy=float(features.energy[frame]),
            onset=float(features.onset[frame]),
            brightness=float(features.brightness[frame]),
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

        prompt = choose_prompt(
            prompt_level=int(controls["prompt_level"]),
            frame=frame,
            total_frames=total_frames,
            beat_pulse=float(features.beat_pulse[frame]),
        )

        do_diffuse = (mode == "full") or (frame in beat_set) or (frame % max(1, cadence) == 0)
        if do_diffuse:
            seed = 42 + frame * 97 + int(controls["seed_jump"])
            current = generator.generate_from_image(
                init_image=transformed,
                prompt=prompt,
                strength=controls["strength"],
                guidance_scale=controls["cfg_scale"],
                seed=seed,
            )
        else:
            current = transformed

        current.save(output_dir / f"frame_{frame:05d}.png")

        if (frame + 1) % 20 == 0 or frame == total_frames - 1:
            beat_tag = "BEAT" if frame in beat_set else "-"
            print(
                f"{frame:>4}/{total_frames-1} [{beat_tag}] "
                f"eng={features.energy[frame]:.2f} onset={features.onset[frame]:.2f} "
                f"str={controls['strength']:.3f} cfg={controls['cfg_scale']:.2f} "
                f"zoom={controls['zoom_delta']:.4f} pan={controls['tx_delta']:+.2f}"
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
        max_seconds=args.max_seconds,
        with_audio=args.with_audio,
    )

