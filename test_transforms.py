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
    
    # Transform parameters
    zoom_per_frame = 0.005  # 0.5% zoom per frame (gradual)
    pan_per_frame = 0.3     # 0.3 pixels right per frame
    
    print(f"\nEffects:")
    print(f"  - Gradual zoom in: {zoom_per_frame*100:.1f}% per frame")
    print(f"  - Slow pan right: {pan_per_frame:.1f}px per frame")
    print(f"  - Total zoom: {1 + zoom_per_frame * TOTAL_FRAMES:.2f}x")
    print(f"  - Total pan: {pan_per_frame * TOTAL_FRAMES:.1f}px\n")
    
    for frame_num in range(1, TOTAL_FRAMES):
        # Calculate cumulative transform
        zoom_factor = 1.0 + (zoom_per_frame * frame_num)
        pan_x = pan_per_frame * frame_num
        
        # Apply transformation to previous frame
        transformed = transform_image(
            current_image,
            zoom=zoom_factor,
            translation_x=pan_x
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
            print(f"   Frame {frame_num}/{TOTAL_FRAMES-1} - Zoom: {zoom_factor:.3f}, Pan: {pan_x:.1f}px")
    
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


if __name__ == "__main__":
    generate_transform_video()
