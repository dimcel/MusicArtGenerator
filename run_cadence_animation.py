#!/usr/bin/env python3
"""
COMPLETE RUNNER: Music-Synced Animation with Optical Flow Cadence

This script demonstrates the full pipeline:
1. Analyze music (beats, amplitude)
2. Generate animation schedule from keyframes  
3. Apply music modifiers to schedule
4. Render with cadence (only diffuse every Nth frame)
5. Use optical flow to interpolate between diffused frames
6. Create final video with audio

USAGE:
    # Test mode (no GPU, just shows schedule):
    python run_cadence_animation.py song.mp3 --test
    
    # Full render:
    python run_cadence_animation.py song.mp3
    
    # Custom cadence (default=4):
    python run_cadence_animation.py song.mp3 --cadence 6
"""

import os
import sys
import argparse
from pathlib import Path


def create_test_audio(output_path: str, duration: float = 10.0, bpm: float = 120.0):
    """Create a test audio file with clear beats"""
    import numpy as np
    from scipy.io import wavfile
    
    sr = 22050
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    
    # Base tone
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    
    # Add beats (kick drum simulation)
    beat_interval = 60.0 / bpm
    for beat_time in np.arange(0, duration, beat_interval):
        beat_start = int(beat_time * sr)
        beat_duration = int(0.1 * sr)
        if beat_start + beat_duration < samples:
            beat_env = np.exp(-np.linspace(0, 5, beat_duration))
            audio[beat_start:beat_start + beat_duration] += 0.7 * beat_env * np.sin(
                2 * np.pi * 60 * np.linspace(0, 0.1, beat_duration)
            )
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    wavfile.write(output_path, sr, (audio * 32767).astype(np.int16))
    print(f"Created test audio: {output_path} ({duration}s, {bpm} BPM)")
    return output_path


def test_mode(audio_path: str, cadence: int = 4, total_frames: int = 120, fps: int = 24):
    """
    Test mode: Show what the schedule would look like without GPU rendering.
    """
    from src.interpolation_engine import Keyframe, AnimationInterpolator
    from src.audio_analyzer import AudioAnalyzer
    from src.music_schedule import MusicSchedule, MusicSyncConfig, apply_music_to_schedule
    from src.optical_flow_cadence import CadenceConfig, CadenceInterpolator, ColorCoherenceMethod
    
    print("\n" + "=" * 70)
    print("🧪 TEST MODE - Showing schedule without GPU rendering")
    print("=" * 70)
    
    # Create test keyframes
    keyframes = [
        Keyframe(
            frame=0,
            prompt="cosmic nebula, vibrant colors, digital art, 8k",
            strength=0.55,
            zoom=1.0,
            seed=42
        ),
        Keyframe(
            frame=total_frames // 2,
            prompt="swirling galaxy, cosmic energy, vibrant, digital art",
            strength=0.6,
            zoom=1.05,
        ),
        Keyframe(
            frame=total_frames - 1,
            prompt="cosmic explosion, supernova, vibrant energy, digital art",
            strength=0.65,
            zoom=1.0,
        ),
    ]
    
    interpolator = AnimationInterpolator(keyframes, total_frames=total_frames)
    
    # Analyze audio
    print(f"\n🎵 Analyzing audio: {audio_path}")
    analyzer = AudioAnalyzer(audio_path)
    analyzer.load()
    
    # Create music schedule
    music = MusicSchedule(analyzer, fps=fps, total_frames=total_frames)
    print(music.summary())
    
    # Configure music sync
    music_config = MusicSyncConfig(
        amplitude_to_strength=(0.5, 0.8),
        beat_strength_boost=0.1,
        beat_zoom_boost=0.05,
    )
    
    # Generate schedules
    base_schedule = interpolator.generate_animation_schedule()
    music_modifiers = music.generate_modifiers(music_config)
    final_schedule = apply_music_to_schedule(base_schedule, music_modifiers)
    
    # Configure cadence
    cadence_config = CadenceConfig(
        cadence=cadence,
        use_optical_flow=True,
        color_coherence=ColorCoherenceMethod.LAB,
    )
    
    cadence_interp = CadenceInterpolator(cadence_config)
    diffusion_frames = cadence_interp.get_diffusion_frames(total_frames)
    
    # Print summary
    print(f"\n📊 CADENCE ANALYSIS")
    print("-" * 70)
    print(f"Total frames:      {total_frames}")
    print(f"Cadence:           {cadence} (diffuse every {cadence}th frame)")
    print(f"Diffusion frames:  {len(diffusion_frames)}")
    print(f"Interpolated:      {total_frames - len(diffusion_frames)}")
    print(f"Speedup:           ~{cadence}x faster!")
    
    # Show schedule sample
    print(f"\n📋 SCHEDULE PREVIEW (showing every {max(1, total_frames//20)} frames)")
    print("-" * 70)
    print(f"{'Frame':>6} {'Type':>10} {'Strength':>10} {'Zoom':>8} {'Beat':>6} {'Amp':>6}")
    print("-" * 70)
    
    step = max(1, total_frames // 20)
    for i in range(0, total_frames, step):
        params = final_schedule[i]
        
        is_diffusion = i in diffusion_frames
        frame_type = "DIFFUSE" if is_diffusion else "interp"
        
        strength = params.get('strength', 0.6)
        if 'strength_boost' in params:
            strength += params['strength_boost']
        
        zoom = params.get('zoom', 1.0) + params.get('zoom_boost', 0.0)
        beat = "🥁" if params.get('is_beat', False) else ""
        amp = params.get('amplitude', 0)
        
        print(f"{i:>6} {frame_type:>10} {strength:>10.3f} {zoom:>8.3f} {beat:>6} {amp:>6.2f}")
    
    print("-" * 70)
    print(f"\n✓ Test complete! Use without --test flag to render.")
    print(f"\n  Render command:")
    print(f"  python {sys.argv[0]} {audio_path} --cadence {cadence}")
    
    return final_schedule


def full_render(
    audio_path: str, 
    cadence: int = 4, 
    total_frames: int = 120, 
    fps: int = 24,
    output_dir: str = "cadence_output",
    width: int = 512,
    height: int = 512,
):
    """
    Full render mode with GPU.
    """
    from src.interpolation_engine import Keyframe, AnimationInterpolator
    from src.sd_animator import StableDiffusionAnimator
    from src.audio_analyzer import AudioAnalyzer
    from src.music_schedule import MusicSchedule, MusicSyncConfig, apply_music_to_schedule
    from src.optical_flow_cadence import CadenceConfig, ColorCoherenceMethod, OpticalFlowMethod
    
    print("\n" + "=" * 70)
    print("🚀 FULL RENDER MODE - With Optical Flow Cadence")
    print("=" * 70)
    
    # Create keyframes
    keyframes = [
        Keyframe(
            frame=0,
            prompt="cosmic nebula, vibrant colors, digital art, 8k",
            strength=0.55,
            zoom=1.0,
            seed=42
        ),
        Keyframe(
            frame=total_frames // 2,
            prompt="swirling galaxy, cosmic energy, vibrant, digital art",
            strength=0.6,
            zoom=1.05,
        ),
        Keyframe(
            frame=total_frames - 1,
            prompt="cosmic explosion, supernova, vibrant energy, digital art",
            strength=0.65,
            zoom=1.0,
        ),
    ]
    
    interpolator = AnimationInterpolator(keyframes, total_frames=total_frames)
    
    # Configure cadence
    cadence_config = CadenceConfig(
        cadence=cadence,
        use_optical_flow=True,
        flow_method=OpticalFlowMethod.DIS_MEDIUM,
        color_coherence=ColorCoherenceMethod.LAB,
        blend_mode="flow",
    )
    
    # Configure music sync
    music_config = MusicSyncConfig(
        amplitude_to_strength=(0.5, 0.8),
        beat_strength_boost=0.1,
        beat_zoom_boost=0.05,
    )
    
    # Initialize animator
    print(f"\n🎨 Initializing Stable Diffusion...")
    animator = StableDiffusionAnimator()
    
    # Render with cadence
    frame_paths = animator.render_with_cadence(
        interpolator=interpolator,
        config=cadence_config,
        output_dir=output_dir,
        width=width,
        height=height,
        num_inference_steps=30,
        fps=fps,
        audio_path=audio_path,
        music_config=music_config,
    )
    
    # Create video
    print(f"\n🎬 Creating video...")
    video_path = os.path.join(output_dir, "animation.mp4")
    animator.create_video(output_dir, video_path, fps=fps)
    
    # Add audio
    print(f"\n🎵 Adding audio...")
    final_video = os.path.join(output_dir, "animation_with_audio.mp4")
    os.system(f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
              f'-c:v copy -c:a aac -shortest "{final_video}" 2>/dev/null')
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print(f"  Frames: {output_dir}/frame_*.png")
    print(f"  Video:  {video_path}")
    print(f"  With audio: {final_video}")
    print("=" * 70)
    
    return frame_paths


def main():
    parser = argparse.ArgumentParser(
        description="Music-synced animation with optical flow cadence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (no GPU):
  python run_cadence_animation.py song.mp3 --test
  
  # Full render with cadence=4 (4x faster):
  python run_cadence_animation.py song.mp3 --cadence 4
  
  # Higher cadence for even faster render:
  python run_cadence_animation.py song.mp3 --cadence 8
        """
    )
    
    parser.add_argument("audio", nargs="?", help="Path to audio file (mp3, wav)")
    parser.add_argument("--test", action="store_true", help="Test mode (no GPU)")
    parser.add_argument("--cadence", type=int, default=4, help="Cadence value (default: 4)")
    parser.add_argument("--frames", type=int, default=120, help="Total frames (default: 120)")
    parser.add_argument("--fps", type=int, default=24, help="FPS (default: 24)")
    parser.add_argument("--output", type=str, default="cadence_output", help="Output directory")
    parser.add_argument("--width", type=int, default=512, help="Frame width")
    parser.add_argument("--height", type=int, default=512, help="Frame height")
    
    args = parser.parse_args()
    
    # Handle audio path
    if args.audio is None:
        print("No audio file provided. Creating test audio...")
        args.audio = "/tmp/test_cadence_audio.wav"
        create_test_audio(args.audio, duration=args.frames / args.fps, bpm=120)
    elif not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        print("Creating test audio instead...")
        args.audio = "/tmp/test_cadence_audio.wav"
        create_test_audio(args.audio, duration=args.frames / args.fps, bpm=120)
    
    if args.test:
        test_mode(
            audio_path=args.audio,
            cadence=args.cadence,
            total_frames=args.frames,
            fps=args.fps,
        )
    else:
        full_render(
            audio_path=args.audio,
            cadence=args.cadence,
            total_frames=args.frames,
            fps=args.fps,
            output_dir=args.output,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()