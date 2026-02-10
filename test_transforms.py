"""
Test Image Transformations
Tests zoom, pan, and rotation effects before diffusion

This shows how to apply Deforum-style transformations:
1. Take previous frame
2. Apply transform (zoom/pan/rotate)
3. Feed into img2img

Current implementation: Approach B - Beat-Intensity Modulation
- Alternating small/big beats control zoom and pan direction
- Small beats: subtle zoom, left pan, medium strength
- Big beats: strong zoom, right pan, high strength

---

APPROACH C - CFG Scale Modulation (Not yet implemented):

What is cfg_scale (Classifier-Free Guidance)?
--------------------------------------------
cfg_scale controls how strictly the AI follows your prompt:

- Low (3-5): More creative/random, less adherent to prompt
- Medium (7-9): Balanced, typical default
- High (10-15): Very strict prompt adherence, less creative freedom

How it would work with beats:
- Small beats: cfg_scale = 7.0 (normal adherence)
- Big beats: cfg_scale = 10.0 (strict adherence)

Result: On strong beats, image follows prompt MORE precisely
        On weak beats, more artistic interpretation

Example usage:
    cfg_scale = 7.0 + (beat_intensity * 3.0)  # Range: 7.0-10.0
    
This creates visual "focus" on beats - the subject becomes more defined
and prominent when strong beats hit, while maintaining flow between.

Combined with strength and seed jumps, this creates triple modulation:
1. strength: how MUCH image changes
2. seed: what DIRECTION it changes
3. cfg_scale: how FOCUSED the change is

---
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _src.image_generator import ImageGenerator, ImageGenerationConfig
from _src.image_transform import transform_image
from _src.frame_interpolator import FrameInterpolator


def generate_fake_beats(total_frames, fps=24, bpm=120):
    """
    Generate fake beat positions and intensities to simulate librosa.
    
    This mimics what MusicSync would return, without needing audio files.
    Real beats have slight variations, so we add some randomness.
    
    Args:
        total_frames: Total video frames
        fps: Frames per second
        bpm: Beats per minute (approximate)
    
    Returns:
        Tuple of (beat_frames, beat_intensities)
        - beat_frames: List of frame numbers where beats occur
        - beat_intensities: List of intensities (0.0-1.0) for each beat
    """
    import random
    random.seed(42)  # Reproducible
    
    # Calculate frames per beat
    frames_per_beat = (60.0 / bpm) * fps
    
    beat_frames = [0]  # Start with frame 0
    beat_intensities = [0.5]  # Medium intensity for first frame
    
    current_frame = 0
    beat_count = 0
    
    while current_frame < total_frames:
        # Add some variation (±10%) to make it realistic
        variation = frames_per_beat * random.uniform(-0.1, 0.1)
        current_frame += int(frames_per_beat + variation)
        
        if current_frame < total_frames:
            beat_frames.append(current_frame)
            
            # Alternate between small (0.3) and big (1.0) beats
            beat_count += 1
            if beat_count % 2 == 1:
                beat_intensities.append(0.3)  # Small beat
            else:
                beat_intensities.append(1.0)  # Big beat
    
    return beat_frames, beat_intensities


def generate_transform_video():
    """
    Generate a video showing zoom and pan effects
    """
    print("\n" + "=" * 70)
    print("🎬 GENERATING TRANSFORMATION VIDEO")
    print("=" * 70)
    
    # Configuration
    PROMPT = "old man sitting on bench, peaceful autumn park, afternoon light"
    TOTAL_FRAMES = 120  # 5 seconds at 24fps
    FPS = 24
    OUTPUT_DIR = Path("transform_video")
    
    print(f"\n📋 Configuration:")
    print(f"   Prompt: {PROMPT}")
    print(f"   Frames: {TOTAL_FRAMES} ({TOTAL_FRAMES/FPS:.1f}s)")
    print(f"   FPS: {FPS}")
    print(f"   Output: {OUTPUT_DIR}/")
    
    # Setup generator
    print(f"\n{'='*70}")
    print("STEP 1: Setup Generator")
    print(f"{'='*70}")
    
    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("✓ Generator ready")
    
    # Generate first frame
    print(f"\n{'='*70}")
    print("STEP 2: Generate Starting Frame")
    print(f"{'='*70}")
    
    current_image = generator.generate_from_text(
        prompt=PROMPT,
        seed=42
    )
    current_image.save(OUTPUT_DIR / "frame_00000.png")
    print("✓ Frame 0 generated")
    
    # Generate animation with transforms
    print(f"\n{'='*70}")
    print("STEP 3: Generate Animation with Transforms")
    print(f"{'='*70}")
    
    # Transform parameters (DELTA per frame, not cumulative!)
    zoom_delta = 1.005  # 0.5% zoom per frame (incremental)
    pan_delta = 0.3     # 0.3 pixels right per frame (incremental)
    
    print(f"\nEffects (Deforum-style incremental):")
    print(f"  - Zoom delta: {(zoom_delta-1)*100:.1f}% per frame")
    print(f"  - Pan delta: {pan_delta:.1f}px per frame")
    print(f"  - Approximate total zoom: {zoom_delta**TOTAL_FRAMES:.2f}x")
    print(f"  - Total pan: ~{pan_delta * TOTAL_FRAMES:.1f}px\n")
    
    for frame_num in range(1, TOTAL_FRAMES):
        # Apply INCREMENTAL transform (delta from previous frame)
        transformed = transform_image(
            current_image,
            zoom=zoom_delta,        # Always 1.005 (not cumulative!)
            translation_x=pan_delta  # Always 0.3px (not cumulative!)
        )
        
        # Generate new frame from transformed image
        current_image = generator.generate_from_image(
            init_image=transformed,
            prompt=PROMPT,
            strength=0.5,  # Medium strength for smooth transitions
            seed=42 + frame_num
        )
        
        # Save frame
        current_image.save(OUTPUT_DIR / f"frame_{frame_num:05d}.png")
        
        # Progress update
        if (frame_num + 1) % 20 == 0 or frame_num == TOTAL_FRAMES - 1:
            print(f"   Frame {frame_num}/{TOTAL_FRAMES-1}")
    
    print(f"\n✓ All frames generated!")
    
    # Create video
    print(f"\n{'='*70}")
    print("STEP 4: Creating Video")
    print(f"{'='*70}")
    
    try:
        from frames_to_video import frames_to_video
        
        video_path = OUTPUT_DIR / "transform_video.mp4"
        frames_to_video(
            frames_dir=str(OUTPUT_DIR),
            output_path=str(video_path),
            fps=FPS
        )
        print(f"\n✓ Video created: {video_path}")
        
    except ImportError:
        print("\n💡 Create video with ffmpeg:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   ffmpeg -framerate {FPS} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p transform_video.mp4")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"\n🎬 Watch the video to see:")
    print(f"   - Gradual zoom in effect")
    print(f"   - Slow pan to the right")
    print(f"   - Smooth img2img transitions")
    print(f"\nOutput: {OUTPUT_DIR}/transform_video.mp4")
    print()


def generate_transform_video_with_interpolation():
    """
    Generate a video with zoom/pan using INTERPOLATION
    
    This generates keyframes with transforms, then interpolates between them.
    Much faster than generating every frame!
    Shows various pan directions (right, left, down, up)
    """
    print("\n" + "=" * 70)
    print("🎬 GENERATING TRANSFORMATION VIDEO (WITH INTERPOLATION)")
    print("=" * 70)
    
    # Configuration
    PROMPT = "old man sitting on bench, peaceful autumn park, afternoon light"
    TOTAL_FRAMES = 120  # 5 seconds at 24fps
    FPS = 24
    BPM = 120  # Fake beats per minute
    OUTPUT_DIR = Path("transform_video_interp")
    
    # Generate fake beats with intensities (simulating what librosa would return)
    beat_frames, beat_intensities = generate_fake_beats(TOTAL_FRAMES, fps=FPS, bpm=BPM)
    
    print(f"\n📋 Configuration:")
    print(f"   Prompt: {PROMPT}")
    print(f"   Total frames: {TOTAL_FRAMES} ({TOTAL_FRAMES/FPS:.1f}s)")
    print(f"   BPM (simulated): {BPM}")
    print(f"   Beat frames: {beat_frames}")
    print(f"   Beat intensities: {['SMALL' if x < 0.5 else 'BIG' for x in beat_intensities]}")
    print(f"   Keyframes to generate: {len(beat_frames)}")
    print(f"   FPS: {FPS}")
    print(f"   Output: {OUTPUT_DIR}/")
    
    # Setup
    print(f"\n{'='*70}")
    print("STEP 1: Setup")
    print(f"{'='*70}")
    
    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    interpolator = FrameInterpolator(method="blend")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("✓ Generator ready")
    print("✓ Interpolator ready")
    
    # Generate keyframes with transforms
    print(f"\n{'='*70}")
    print("STEP 2: Generate Keyframes with Transforms")
    print(f"{'='*70}")
    
    # Generate keyframes at beat positions WITH VARIABLE TRANSFORMS
    print(f"\n🎵 Beat-Intensity Modulation (Approach B):")
    print(f"   Small beats (0.3): subtle zoom, small left pan")
    print(f"   Big beats (1.0): strong zoom, small right pan")
    print()
    
    keyframe_images = {}  # {frame_num: image}
    
    # Generate first frame
    current_image = generator.generate_from_text(
        prompt=PROMPT,
        seed=42
    )
    keyframe_images[0] = current_image
    print(f"   ✓ Initial frame 0 generated")
    
    for i in range(1, len(beat_frames)):
        prev_beat = beat_frames[i - 1]
        curr_beat = beat_frames[i]
        frames_between = curr_beat - prev_beat
        beat_intensity = beat_intensities[i]
        
        # Calculate transforms based on beat intensity
        # Small beat (0.3): subtle zoom, pan left
        # Big beat (1.0): strong zoom, pan right
        if beat_intensity < 0.5:
            # SMALL BEAT
            zoom_delta = 1.008   # 0.8% zoom per frame
            pan_x_delta = -0.3   # Small left pan
            angle_delta = -0.1   # Small rotation
            strength = 0.65      # Medium-low strength
            beat_label = "SMALL"
        else:
            # BIG BEAT
            zoom_delta = 1.020   # 2.0% zoom per frame
            pan_x_delta = 0.5    # Small right pan
            angle_delta = -0.3   # Stronger rotation
            strength = 0.90      # High strength
            beat_label = "BIG"
        
        # Apply transform for each frame between beats
        for _ in range(frames_between):
            current_image = transform_image(
                current_image,
                zoom=zoom_delta,
                angle=angle_delta,
                translation_x=pan_x_delta
            )
        
        # Generate new keyframe at beat position
        current_image = generator.generate_from_image(
            init_image=current_image,
            prompt=PROMPT,
            strength=strength,  # Varies by beat intensity!
            seed=42 + i * 137  # Seed jump for variety
            # NOTE: cfg_scale would go here for Approach C
            # cfg_scale=7.0 + (beat_intensity * 3.0)  # 7.0-10.0 range
            # Higher cfg_scale = more prompt adherence on strong beats
        )
        
        keyframe_images[curr_beat] = current_image
        print(f"   ✓ {beat_label} beat @ frame {curr_beat} (zoom={zoom_delta:.3f}, pan={pan_x_delta:+.1f}px, strength={strength:.2f})")
    
    print(f"\n✓ Generated {len(keyframe_images)} beat keyframes")
    
    # Interpolate between keyframes
    print(f"\n{'='*70}")
    print("STEP 3: Interpolate Between Keyframes")
    print(f"{'='*70}")
    
    all_frames = []
    
    for i in range(len(beat_frames) - 1):
        frame_a_num = beat_frames[i]
        frame_b_num = beat_frames[i + 1]
        
        frame_a = keyframe_images[frame_a_num]
        frame_b = keyframe_images[frame_b_num]
        
        # Add keyframe A
        all_frames.append(frame_a)
        
        # Interpolate between
        num_between = frame_b_num - frame_a_num - 1
        if num_between > 0:
            interpolated = interpolator.interpolate(frame_a, frame_b, num_between)
            all_frames.extend(interpolated)
            print(f"   Interpolated {num_between} frames between {frame_a_num} and {frame_b_num}")
    
    # Add final keyframe
    all_frames.append(keyframe_images[beat_frames[-1]])
    
    print(f"\n✓ Total frames: {len(all_frames)}")
    
    # Save all frames
    print(f"\n{'='*70}")
    print("STEP 4: Saving Frames")
    print(f"{'='*70}")
    
    for i, frame in enumerate(all_frames):
        frame.save(OUTPUT_DIR / f"frame_{i:05d}.png")
        if (i + 1) % 20 == 0 or i == len(all_frames) - 1:
            print(f"   Saved {i+1}/{len(all_frames)} frames")
    
    # Create video
    print(f"\n{'='*70}")
    print("STEP 5: Creating Video")
    print(f"{'='*70}")
    
    try:
        from frames_to_video import frames_to_video
        
        video_path = OUTPUT_DIR / "transform_beat_sync.mp4"
        frames_to_video(
            frames_dir=str(OUTPUT_DIR),
            output_path=str(video_path),
            fps=FPS
        )
        print(f"\n✓ Video created: {video_path}")
        
    except ImportError:
        print("\n💡 Create video with ffmpeg:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   ffmpeg -framerate {FPS} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p transform_beat_sync.mp4")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    
    savings = (1 - len(beat_frames) / TOTAL_FRAMES) * 100
    print(f"\n📊 Statistics:")
    print(f"   Beat keyframes generated: {len(beat_frames)}")
    print(f"   Frames interpolated: {len(all_frames) - len(beat_frames)}")
    print(f"   Total frames: {len(all_frames)}")
    print(f"   💰 Cost savings: {savings:.1f}% fewer generations!")
    print(f"\n🎵 BEAT-INTENSITY MODULATION (Approach B):")
    print(f"   - Small beats: subtle zoom (0.8%), left pan, medium strength (0.65)")
    print(f"   - Big beats: strong zoom (2.0%), right pan, high strength (0.90)")
    print(f"   - Alternating pattern creates rhythm in visuals")
    print(f"   - Pan direction changes with beat intensity!")
    print(f"\n🎬 Watch the video to see:")
    print(f"   - Visual 'pops' synchronized with fake beats")
    print(f"   - DIFFERENT zoom/pan on small vs big beats")
    print(f"   - Left drift on small beats, right drift on big beats")
    print(f"   - Rotation (counterclockwise turn)")
    print(f"\nOutput: {OUTPUT_DIR}/transform_beat_sync.mp4")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interp":
        generate_transform_video_with_interpolation()
    else:
        print("\nChoose a test:")
        print("  1. Full generation (120 frames generated)")
        print("  2. With interpolation (10 keyframes + interpolation)")
        print()
        choice = input("Enter 1 or 2: ").strip()
        
        if choice == "1":
            generate_transform_video()
        elif choice == "2":
            generate_transform_video_with_interpolation()
        else:
            print("Running interpolation test by default...")
            generate_transform_video_with_interpolation()
