"""
Test Music Sync - Strategy 1: Beat-Triggered Keyframes

This test demonstrates simple music synchronization:
1. Load audio and detect beats
2. Generate keyframes at beat positions
3. Use higher strength on beats for more dramatic changes
4. Interpolate between keyframes

Inspired by sd-parseq's beat-locked keyframe scheduling.
"""

from pathlib import Path

from music_art_generator.music_sync import MusicSync, simple_beat_detection


def test_beat_detection_only():
    """
    Test 1: Just detect beats from an audio file.
    
    This is the simplest test - load audio and see what beats are detected.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Simple Beat Detection")
    print("=" * 70)
    
    # You'll need to provide your own audio file
    # For testing, use any mp3/wav file
    audio_path = "test_audio.mp3"  # CHANGE THIS to your audio file
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"\n⚠️  Audio file not found: {audio_path}")
        print("   Please update audio_path in this test file to point to a real audio file.")
        print("   Supported formats: mp3, wav, flac, ogg, etc.")
        return None
    
    # Simple approach: one function call
    print("\n📊 Using simple_beat_detection():")
    schedule = simple_beat_detection(audio_path, fps=24)
    print(f"   BPM: {schedule.bpm:.1f}")
    print(f"   Beats detected: {len(schedule.beat_frames)}")
    print(f"   First 10 beat frames: {schedule.beat_frames[:10]}")
    
    return schedule


def test_music_sync_class():
    """
    Test 2: Use MusicSync class with full parameter control.
    
    This shows how to use the class for more control over strength values.
    """
    print("\n" + "=" * 70)
    print("TEST 2: MusicSync Class")
    print("=" * 70)
    
    audio_path = "test_audio.mp3"  # CHANGE THIS
    
    if not Path(audio_path).exists():
        print(f"\n⚠️  Audio file not found: {audio_path}")
        return None
    
    # Create sync with custom strength values
    sync = MusicSync(
        audio_path=audio_path,
        fps=24,
        beat_strength=0.85,    # High change on beats
        normal_strength=0.4    # Smooth between beats
    )
    
    # Load and analyze
    sync.load()
    
    # Print schedule
    sync.print_schedule(max_display=15)
    
    return sync


def test_parameter_modulation(sync: MusicSync):
    """
    Test 3: Show how parameters change frame-by-frame.
    
    This demonstrates the core of Strategy 1: modulating generation
    parameters based on beat timing.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Parameter Modulation (Strategy 1)")
    print("=" * 70)
    
    if sync is None:
        print("⚠️  Sync not initialized, skipping test")
        return
    
    print("\nShowing how strength and seed change frame-by-frame:\n")
    print(f"{'Frame':<8} {'On Beat?':<12} {'Strength':<12} {'Seed Offset':<15}")
    print("-" * 60)
    
    # Show first 50 frames
    base_seed = 42
    for frame in range(50):
        is_beat = sync.is_beat_frame(frame)
        strength = sync.get_strength_for_frame(frame)
        seed_offset = sync.get_seed_offset_for_frame(frame, base_seed)
        
        marker = "🎵 BEAT!" if is_beat else ""
        print(f"{frame:<8} {str(is_beat):<12} {strength:<12.2f} {seed_offset:<15} {marker}")
    
    print("\n✓ Notice how:")
    print("  - Strength is higher (0.85) on beat frames")
    print("  - Strength is lower (0.40) between beats")
    print("  - Seed jumps significantly on beats for variation")


def test_keyframe_schedule(sync: MusicSync):
    """
    Test 4: Generate keyframe schedule for video generation.
    
    This shows which frames should be generated vs interpolated.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Keyframe Schedule")
    print("=" * 70)
    
    if sync is None:
        print("⚠️  Sync not initialized, skipping test")
        return
    
    # Get keyframe schedule for first 5 seconds at 24fps
    max_frames = 5 * 24  # 5 seconds
    keyframes = sync.get_keyframe_schedule(max_frames=max_frames)
    
    print(f"\nFor a {max_frames} frame video ({max_frames/24:.1f}s):")
    print(f"  - Keyframes to generate: {len(keyframes)}")
    print(f"  - Frames to interpolate: {max_frames - len(keyframes)}")
    print(f"  - Keyframe positions: {keyframes}")
    
    print("\n💡 Generation strategy:")
    print("  1. Generate frames at these keyframe positions")
    print("  2. Use higher strength on beat keyframes")
    print("  3. Interpolate all frames in between")
    print("  4. This reduces generation cost while keeping beat sync!")
    
    # Calculate savings
    interpolation_savings = (1 - len(keyframes) / max_frames) * 100
    print(f"\n💰 Cost savings: {interpolation_savings:.1f}% fewer generations needed")


def test_example_workflow():
    """
    Test 5: Complete example workflow for music-synced video.
    
    This is a blueprint for how you'd use this in video_animator.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Example Workflow (Blueprint)")
    print("=" * 70)
    
    print("""
Example workflow for music-synced video generation:

```python
from music_art_generator.music_sync import MusicSync
from music_art_generator.video_animator import VideoAnimator
from music_art_generator.frame_interpolator import FrameInterpolator

# 1. Set up music sync
sync = MusicSync("song.mp3", fps=24)
sync.load()

# 2. Get keyframe schedule
keyframes = sync.get_keyframe_schedule(max_frames=240)  # 10 seconds

# 3. Generate only keyframes
generated_frames = {}
current_image = None

for frame_num in keyframes:
    # Get beat-modulated strength
    strength = sync.get_strength_for_frame(frame_num)
    seed = 42 + sync.get_seed_offset_for_frame(frame_num, 42)
    
    # Generate frame
    if frame_num == 0:
        current_image = generator.generate_from_text(prompt, seed)
    else:
        current_image = generator.generate_from_image(
            current_image, 
            prompt, 
            strength=strength,  # Higher on beats!
            seed=seed
        )
    
    generated_frames[frame_num] = current_image
    print(f"Generated frame {frame_num} (strength={strength:.2f})")

# 4. Interpolate between keyframes
interpolator = FrameInterpolator(method="optical_flow")
all_frames = []

for i in range(len(keyframes) - 1):
    frame_a_num = keyframes[i]
    frame_b_num = keyframes[i + 1]
    
    frame_a = generated_frames[frame_a_num]
    frame_b = generated_frames[frame_b_num]
    
    # Add first keyframe
    all_frames.append(frame_a)
    
    # Interpolate between
    num_between = frame_b_num - frame_a_num - 1
    if num_between > 0:
        interpolated = interpolator.interpolate(frame_a, frame_b, num_between)
        all_frames.extend(interpolated)

# Add final frame
all_frames.append(generated_frames[keyframes[-1]])

# 5. Create video
# ... save frames and run ffmpeg ...
```

Key benefits of this approach:
✓ Fewer generations (only at keyframes)
✓ Visual changes synchronized with beats
✓ Smooth motion between beats (interpolation)
✓ Control over how dramatic the beat changes are
""")


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "=" * 70)
    print("🎵 MUSIC SYNC TESTS - Strategy 1: Beat-Triggered Keyframes")
    print("=" * 70)
    
    # Test 1: Simple detection
    schedule = test_beat_detection_only()
    
    # Test 2: Full class usage
    sync = test_music_sync_class()
    
    # Test 3: Parameter modulation
    test_parameter_modulation(sync)
    
    # Test 4: Keyframe schedule
    test_keyframe_schedule(sync)
    
    # Test 5: Example workflow
    test_example_workflow()
    
    print("\n" + "=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("  1. Get an audio file (mp3, wav, etc.)")
    print("  2. Update audio_path in this test file")
    print("  3. Run: python test_music_sync.py")
    print("  4. Once working, integrate into video_animator.py")
    print("\n")


if __name__ == "__main__":
    run_all_tests()
