"""
Test Image Transformations
Tests zoom, pan, and rotation effects before diffusion

This shows how to apply Deforum-style transformations:
1. Take previous frame
2. Apply transform (zoom/pan/rotate)
3. Feed into img2img
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _src.image_generator import ImageGenerator, ImageGenerationConfig
from _src.image_transform import transform_image
from _src.frame_interpolator import FrameInterpolator


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
    KEYFRAME_INTERVAL = 12  # Generate every 12th frame
    FPS = 24
    OUTPUT_DIR = Path("transform_video_interp")
    
    print(f"\n📋 Configuration:")
    print(f"   Prompt: {PROMPT}")
    print(f"   Total frames: {TOTAL_FRAMES} ({TOTAL_FRAMES/FPS:.1f}s)")
    print(f"   Keyframe every: {KEYFRAME_INTERVAL} frames")
    print(f"   Keyframes to generate: {len(range(0, TOTAL_FRAMES, KEYFRAME_INTERVAL))}")
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
    
    # Small transforms (subtle effects)
    # zoom_delta = 1.005  # 0.5% zoom per frame
    # pan_x_delta = 0.5   # 0.5 pixels right per frame
    # angle_delta = -0.02  # Small left turn
    
    # Strong transforms (dramatic effects)
    zoom_delta = 1.015  # 1.5% zoom per frame - much more dramatic!
    pan_x_delta = 1.0   # 1 pixel right per frame
    angle_delta = -0.2  # Strong left rotation
    
    print(f"\nEffects:")
    print(f"  - Zoom: {(zoom_delta-1)*100:.1f}% per frame")
    print(f"  - Pan: {pan_x_delta:.1f}px right per frame")
    print(f"  - Rotation: {angle_delta:.2f}° per frame (counterclockwise)")
    print(f"  - Total pan: ~{pan_x_delta * TOTAL_FRAMES:.1f}px to the right")
    print(f"  - Total rotation: ~{angle_delta * TOTAL_FRAMES:.1f}°")
    
    keyframes = {}  # {frame_num: image}
    current_image = None
    
    # Generate first frame
    current_image = generator.generate_from_text(
        prompt=PROMPT,
        seed=42
    )
    keyframes[0] = current_image
    print(f"   ✓ Keyframe 0 generated")
    
    # Generate keyframes at intervals
    keyframe_nums = list(range(KEYFRAME_INTERVAL, TOTAL_FRAMES, KEYFRAME_INTERVAL))
    
    for i, frame_num in enumerate(keyframe_nums, start=1):
        # Apply transform KEYFRAME_INTERVAL times (zoom + pan + rotate)
        for _ in range(KEYFRAME_INTERVAL):
            current_image = transform_image(
                current_image,
                zoom=zoom_delta,
                angle=angle_delta,
                translation_x=pan_x_delta
            )
        
        # Generate new keyframe
        current_image = generator.generate_from_image(
            init_image=current_image,
            prompt=PROMPT,
            strength=0.5,
            seed=42 + frame_num
        )
        
        keyframes[frame_num] = current_image
        print(f"   ✓ Keyframe {frame_num} generated ({i}/{len(keyframe_nums)})")
    
    print(f"\n✓ Generated {len(keyframes)} keyframes")
    
    # Interpolate between keyframes
    print(f"\n{'='*70}")
    print("STEP 3: Interpolate Between Keyframes")
    print(f"{'='*70}")
    
    all_frames = []
    keyframe_list = sorted(keyframes.keys())
    
    for i in range(len(keyframe_list) - 1):
        frame_a_num = keyframe_list[i]
        frame_b_num = keyframe_list[i + 1]
        
        frame_a = keyframes[frame_a_num]
        frame_b = keyframes[frame_b_num]
        
        # Add keyframe A
        all_frames.append(frame_a)
        
        # Interpolate between
        num_between = frame_b_num - frame_a_num - 1
        if num_between > 0:
            interpolated = interpolator.interpolate(frame_a, frame_b, num_between)
            all_frames.extend(interpolated)
            print(f"   Interpolated {num_between} frames between {frame_a_num} and {frame_b_num}")
    
    # Add final keyframe
    all_frames.append(keyframes[keyframe_list[-1]])
    
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
        
        video_path = OUTPUT_DIR / "transform_video_interp.mp4"
        frames_to_video(
            frames_dir=str(OUTPUT_DIR),
            output_path=str(video_path),
            fps=FPS
        )
        print(f"\n✓ Video created: {video_path}")
        
    except ImportError:
        print("\n💡 Create video with ffmpeg:")
        print(f"   cd {OUTPUT_DIR}")
        print(f"   ffmpeg -framerate {FPS} -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p transform_video_interp.mp4")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    
    savings = (1 - len(keyframes) / TOTAL_FRAMES) * 100
    print(f"\n📊 Statistics:")
    print(f"   Keyframes generated: {len(keyframes)}")
    print(f"   Frames interpolated: {len(all_frames) - len(keyframes)}")
    print(f"   Total frames: {len(all_frames)}")
    print(f"   💰 Cost savings: {savings:.1f}% fewer generations!")
    print(f"\n🎬 Watch the video to see:")
    print(f"   - Strong zoom in with interpolation")
    print(f"   - Rotation (counterclockwise turn)")
    print(f"   - Pan to the right")
    print(f"   - Much faster generation!")
    print(f"\nOutput: {OUTPUT_DIR}/transform_video_interp.mp4")
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
