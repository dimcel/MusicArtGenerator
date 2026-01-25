"""
Run Music-Synced Animation
Complete example script for generating animations synchronized to music

Usage:
    python run_music_animation.py

Requirements:
    - interpolation_engine.py
    - sd_animator.py (with the 2 new methods added)
    - audio_analyzer.py
    - music_schedule.py
    - A music file (mp3, wav, etc.)
    - GPU with CUDA (or CPU, but very slow)
"""

import os
import sys
from interpolation_engine import Keyframe, AnimationInterpolator
from music_schedule import MusicSyncConfig

# Only import SD animator if not in test mode (requires GPU)
if "--test" not in sys.argv:
    from sd_animator import StableDiffusionAnimator


def run_music_animation(
    audio_path: str,
    output_dir: str = "music_video",
    total_frames: int = 120,
    fps: int = 24,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 25
):
    """
    Generate a music-synced animation.
    
    Args:
        audio_path: Path to your music file (mp3, wav, etc.)
        output_dir: Directory to save output frames
        total_frames: Number of frames to generate
        fps: Frames per second
        width: Frame width
        height: Frame height
        num_inference_steps: Denoising steps (lower = faster, higher = better quality)
    """
    
    print("=" * 60)
    print("🎬 MUSIC-SYNCED ANIMATION GENERATOR")
    print("=" * 60)
    
    # =========================================================================
    # 1. DEFINE YOUR KEYFRAMES
    # =========================================================================
    # Edit these prompts to whatever you want!
    
    keyframes = [
        Keyframe(
            frame=0,
            prompt="a cosmic nebula in deep space, vibrant purple and blue colors, stars, photorealistic",
            negative_prompt="blurry, low quality, text, watermark",
            strength=0.55,
            zoom=1.0,
            seed=42  # Fixed seed for reproducibility
        ),
        Keyframe(
            frame=total_frames // 3,
            prompt="a cosmic nebula with swirling galaxies, golden and orange energy, stars exploding",
            negative_prompt="blurry, low quality, text, watermark",
            strength=0.6,
            zoom=1.05,
        ),
        Keyframe(
            frame=(total_frames * 2) // 3,
            prompt="a cosmic explosion, supernova, intense bright light, energy waves",
            negative_prompt="blurry, low quality, text, watermark",
            strength=0.65,
            zoom=1.1,
        ),
        Keyframe(
            frame=total_frames - 1,
            prompt="calm space after explosion, scattered stardust, peaceful nebula, deep blue",
            negative_prompt="blurry, low quality, text, watermark",
            strength=0.55,
            zoom=1.0,
        ),
    ]
    
    print(f"\n📍 Keyframes defined: {len(keyframes)}")
    for kf in keyframes:
        print(f"   Frame {kf.frame}: {kf.prompt[:50]}...")
    
    # =========================================================================
    # 2. CREATE INTERPOLATOR
    # =========================================================================
    
    interpolator = AnimationInterpolator(keyframes, total_frames=total_frames)
    print(f"\n🎞️  Total frames: {total_frames}")
    print(f"   Duration: {total_frames / fps:.1f} seconds at {fps} FPS")
    
    # =========================================================================
    # 3. CONFIGURE MUSIC SYNC
    # =========================================================================
    # Adjust these values to change how music affects the animation
    
    music_config = MusicSyncConfig(
        # Amplitude (loudness) affects strength
        # When quiet: strength = 0.5, when loud: strength = 0.75
        amplitude_to_strength=(0.5, 0.75),
        
        # On beat frames, add extra strength (more change)
        beat_strength_boost=0.1,
        
        # On beat frames, zoom in slightly
        beat_zoom_boost=0.03,
        
        # Smooth amplitude over 2 frames to avoid jitter
        amplitude_smoothing=2,
        
        # How close a frame needs to be to a beat (in seconds)
        beat_tolerance=0.05,
    )
    
    print(f"\n🎵 Music sync config:")
    print(f"   Amplitude → Strength: {music_config.amplitude_to_strength}")
    print(f"   Beat strength boost: +{music_config.beat_strength_boost}")
    print(f"   Beat zoom boost: +{music_config.beat_zoom_boost}")
    
    # =========================================================================
    # 4. CREATE ANIMATOR AND RENDER
    # =========================================================================
    
    print(f"\n🚀 Initializing Stable Diffusion...")
    from sd_animator import StableDiffusionAnimator
    animator = StableDiffusionAnimator()
    
    print(f"\n🎬 Starting render with music: {audio_path}")
    frame_paths = animator.render_animation_with_music(
        interpolator,
        audio_path=audio_path,
        music_config=music_config,
        output_dir=output_dir,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        fps=fps
    )
    
    # =========================================================================
    # 5. CREATE VIDEO
    # =========================================================================
    
    video_path = os.path.join(output_dir, "animation.mp4")
    print(f"\n🎥 Creating video: {video_path}")
    animator.create_video(output_dir, video_path, fps=fps)
    
    # =========================================================================
    # 6. OPTIONALLY ADD AUDIO TO VIDEO
    # =========================================================================
    
    final_video = os.path.join(output_dir, "animation_with_audio.mp4")
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            final_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Video with audio: {final_video}")
    except Exception as e:
        print(f"⚠ Could not add audio to video: {e}")
        print(f"  Video without audio is at: {video_path}")
    
    print("\n" + "=" * 60)
    print("✅ ANIMATION COMPLETE!")
    print("=" * 60)
    print(f"   Frames: {output_dir}/")
    print(f"   Video: {video_path}")
    
    return frame_paths


# =============================================================================
# ALTERNATIVE: Test without GPU (schedule only)
# =============================================================================

def test_schedule_only(audio_path: str):
    """
    Test the music schedule without rendering (no GPU needed).
    Useful for checking beat detection and amplitude mapping.
    """
    from audio_analyzer import AudioAnalyzer
    from music_schedule import MusicSchedule, MusicSyncConfig, apply_music_to_schedule
    
    print("=" * 60)
    print("🧪 TEST MODE: Schedule Generation Only (No GPU)")
    print("=" * 60)
    
    # Simple keyframes
    keyframes = [
        Keyframe(frame=0, prompt="test A", strength=0.5, zoom=1.0),
        Keyframe(frame=60, prompt="test B", strength=0.6, zoom=1.1),
        Keyframe(frame=119, prompt="test C", strength=0.5, zoom=1.0),
    ]
    
    interpolator = AnimationInterpolator(keyframes, total_frames=120)
    
    # Analyze audio
    print(f"\n🎵 Analyzing: {audio_path}")
    analyzer = AudioAnalyzer(audio_path)
    analyzer.load()
    
    # Create schedule
    music = MusicSchedule(analyzer, fps=24, total_frames=120)
    print(music.summary())
    
    # Config
    config = MusicSyncConfig(
        amplitude_to_strength=(0.5, 0.8),
        beat_zoom_boost=0.05,
        beat_strength_boost=0.1,
    )
    
    # Generate
    base_schedule = interpolator.generate_animation_schedule()
    modifiers = music.generate_modifiers(config)
    final_schedule = apply_music_to_schedule(base_schedule, modifiers)
    
    # Show results
    print("\n📊 Schedule Preview (every 10 frames):")
    print("-" * 70)
    print(f"{'Frame':>6} {'Amplitude':>10} {'Strength':>10} {'Zoom':>8} {'Beat':>6}")
    print("-" * 70)
    
    for i in range(0, 120, 10):
        f = final_schedule[i]
        amp = f.get('amplitude', 0)
        strength = f.get('strength', 0)
        if 'strength_boost' in f:
            strength += f['strength_boost']
        zoom = f.get('zoom', 1.0) + f.get('zoom_boost', 0)
        beat = "🥁" if f.get('is_beat') else ""
        
        print(f"{i:>6} {amp:>10.3f} {strength:>10.3f} {zoom:>8.3f} {beat:>6}")
    
    # Count beats
    beat_count = sum(1 for f in final_schedule if f.get('is_beat'))
    print(f"\nTotal beat frames: {beat_count}")
    
    return final_schedule


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for audio file argument
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_music_animation.py <audio_file.mp3>")
        print("  python run_music_animation.py <audio_file.mp3> --test")
        print("")
        print("Examples:")
        print("  python run_music_animation.py my_song.mp3")
        print("  python run_music_animation.py my_song.mp3 --test  # No GPU needed")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Test mode (no GPU)
    if "--test" in sys.argv:
        test_schedule_only(audio_path)
    else:
        # Full render
        run_music_animation(
            audio_path=audio_path,
            output_dir="music_video",
            total_frames=120,      # 5 seconds at 24fps
            fps=24,
            width=512,
            height=512,
            num_inference_steps=25  # Lower = faster, higher = better quality
        )