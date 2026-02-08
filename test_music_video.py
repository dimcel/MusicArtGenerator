"""
Simple Music-Synced Video Generator
Generates a short video (5 seconds) synchronized to music beats

Usage:
    python test_music_video.py
    
Requirements:
    - Audio file (update AUDIO_PATH below)
    - librosa installed
"""

import sys
import os
from pathlib import Path

# Add _src to path
sys.path.insert(0, str(Path(__file__).parent))

from _src.music_sync import MusicSync
from _src.image_generator import ImageGenerator, ImageGenerationConfig
from _src.frame_interpolator import FrameInterpolator


# =============================================================================
# CONFIGURATION - Edit these
# =============================================================================
AUDIO_PATH = "test_audio.mp3"  # YOUR AUDIO FILE HERE
PROMPT = "old man sitting on bench, autumn park, peaceful"
OUTPUT_DIR = "music_video_test"
DURATION_SECONDS = 5  # Keep short for testing
FPS = 24
# =============================================================================


def generate_music_synced_video():
    """
    Generate a simple music-synced video.
    
    Process:
    1. Detect beats from audio
    2. Generate keyframes at beat positions
    3. Interpolate frames between beats
    4. Save video
    """
    
    print("\n" + "=" * 70)
    print("🎵 MUSIC-SYNCED VIDEO GENERATOR")
    print("=" * 70)
    
    # Check audio file
    if not Path(AUDIO_PATH).exists():
        print(f"\n❌ Audio file not found: {AUDIO_PATH}")
        print("   Update AUDIO_PATH in this script to point to your audio file")
        return
    
    # Setup
    print(f"\n📋 Configuration:")
    print(f"   Audio: {AUDIO_PATH}")
    print(f"   Prompt: {PROMPT}")
    print(f"   Duration: {DURATION_SECONDS}s")
    print(f"   FPS: {FPS}")
    print(f"   Output: {OUTPUT_DIR}/")
    
    # Step 1: Load music and detect beats
    print(f"\n{'='*70}")
    print("STEP 1: Analyzing Music")
    print(f"{'='*70}")
    
    sync = MusicSync(
        audio_path=AUDIO_PATH,
        fps=FPS,
        beat_strength=0.85,    # High change on beats
        normal_strength=0.4     # Smooth between beats
    )
    sync.load()
    
    # Get keyframe schedule
    max_frames = DURATION_SECONDS * FPS
    keyframes = sync.get_keyframe_schedule(max_frames=max_frames)
    
    print(f"\n✓ Detected {len(keyframes)} keyframes to generate")
    print(f"✓ Will interpolate {max_frames - len(keyframes)} frames")
    print(f"✓ Total frames: {max_frames}")
    
    # Step 2: Setup image generator
    print(f"\n{'='*70}")
    print("STEP 2: Setting Up Generator")
    print(f"{'='*70}")
    
    gen_config = ImageGenerationConfig(
        width=512,
        height=512,
        negative_prompt="blurry, bad quality"
    )
    generator = ImageGenerator(gen_config)
    print("✓ Generator ready")
    
    # Step 3: Generate keyframes
    print(f"\n{'='*70}")
    print("STEP 3: Generating Keyframes")
    print(f"{'='*70}")
    
    generated_frames = {}
    current_image = None
    base_seed = 42
    
    for i, frame_num in enumerate(keyframes):
        is_beat = sync.is_beat_frame(frame_num)
        strength = sync.get_strength_for_frame(frame_num)
        seed = base_seed + sync.get_seed_offset_for_frame(frame_num, base_seed)
        
        print(f"\n[{i+1}/{len(keyframes)}] Frame {frame_num}")
        print(f"   Beat: {is_beat}, Strength: {strength:.2f}, Seed: {seed}")
        
        if frame_num == 0:
            # First frame from text
            current_image = generator.generate_from_text(
                prompt=PROMPT,
                seed=seed
            )
            print("   ✓ Generated from text")
        else:
            # Subsequent frames from previous
            current_image = generator.generate_from_image(
                init_image=current_image,
                prompt=PROMPT,
                strength=strength,
                seed=seed
            )
            print("   ✓ Generated from previous")
        
        generated_frames[frame_num] = current_image
    
    print(f"\n✓ Generated {len(generated_frames)} keyframes")
    
    # Step 4: Interpolate between keyframes
    print(f"\n{'='*70}")
    print("STEP 4: Interpolating Frames")
    print(f"{'='*70}")
    
    interpolator = FrameInterpolator(method="blend")  # Simple blend
    all_frames = []
    
    for i in range(len(keyframes) - 1):
        frame_a_num = keyframes[i]
        frame_b_num = keyframes[i + 1]
        
        frame_a = generated_frames[frame_a_num]
        frame_b = generated_frames[frame_b_num]
        
        # Add keyframe
        all_frames.append(frame_a)
        
        # Interpolate between
        num_between = frame_b_num - frame_a_num - 1
        if num_between > 0:
            print(f"   Interpolating {num_between} frames between {frame_a_num} and {frame_b_num}")
            interpolated = interpolator.interpolate(frame_a, frame_b, num_between)
            all_frames.extend(interpolated)
    
    # Add final keyframe
    all_frames.append(generated_frames[keyframes[-1]])
    
    print(f"\n✓ Total frames: {len(all_frames)}")
    
    # Step 5: Save frames
    print(f"\n{'='*70}")
    print("STEP 5: Saving Frames")
    print(f"{'='*70}")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    for i, frame in enumerate(all_frames):
        frame_path = output_path / f"frame_{i:05d}.png"
        frame.save(frame_path)
        if (i + 1) % 20 == 0 or i == len(all_frames) - 1:
            print(f"   Saved {i+1}/{len(all_frames)} frames")
    
    print(f"\n✓ All frames saved to {OUTPUT_DIR}/")
    
    # Step 6: Create video
    print(f"\n{'='*70}")
    print("STEP 6: Creating Video")
    print(f"{'='*70}")
    
    video_no_audio = output_path / "video_no_audio.mp4"
    video_path = output_path / "music_video.mp4"
    
    # Create video without audio first
    try:
        from frames_to_video import frames_to_video
        
        frames_to_video(
            frames_dir=str(output_path),
            output_path=str(video_no_audio),
            fps=FPS
        )
        print(f"✓ Video created (no audio): {video_no_audio}")
        
        # Add audio to video
        print(f"\n   Adding audio...")
        audio_cmd = (
            f'ffmpeg -y -i "{video_no_audio}" -i "{AUDIO_PATH}" '
            f'-c:v copy -c:a aac -shortest "{video_path}"'
        )
        result = os.system(audio_cmd)
        
        if result == 0:
            print(f"✓ Final video with audio: {video_path}")
            # Clean up video without audio
            video_no_audio.unlink()
        else:
            print(f"⚠️  Could not add audio, but video saved: {video_no_audio}")
        
    except ImportError:
        # Manual ffmpeg command
        print("\n💡 Create video with ffmpeg:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   ffmpeg -framerate {FPS} -i frame_%05d.png -i ../{AUDIO_PATH} \\")
        print(f"          -c:v libx264 -pix_fmt yuv420p -shortest \\")
        print(f"          -c:a aac music_video.mp4")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Keyframes generated: {len(keyframes)}")
    print(f"   Frames interpolated: {len(all_frames) - len(keyframes)}")
    print(f"   Total frames: {len(all_frames)}")
    print(f"   Duration: {len(all_frames) / FPS:.1f}s")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"\n🎵 The video should show dramatic changes on beats!")
    print()


if __name__ == "__main__":
    generate_music_synced_video()
