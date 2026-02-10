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
from _src.image_transform import transform_image, zoom_in, pan_right


def test_basic_transformations():
    """
    Test 1: Basic transformation functions
    """
    print("\n" + "=" * 70)
    print("TEST 1: Basic Image Transformations")
    print("=" * 70)
    
    # Generate a test image
    print("\n1. Generating test image...")
    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    
    image = generator.generate_from_text(
        prompt="old man on bench, autumn park",
        seed=42
    )
    print("   ✓ Generated base image")
    
    # Create output directory
    output_dir = Path("test_transforms")
    output_dir.mkdir(exist_ok=True)
    
    # Save original
    image.save(output_dir / "00_original.png")
    print("\n2. Applying transformations...")
    
    # Test zoom
    zoomed = zoom_in(image, amount=1.1)
    zoomed.save(output_dir / "01_zoom_in_10pct.png")
    print("   ✓ Zoom in 10%")
    
    # Test pan
    panned = pan_right(image, pixels=20)
    panned.save(output_dir / "02_pan_right_20px.png")
    print("   ✓ Pan right 20 pixels")
    
    # Test combined
    combined = transform_image(image, zoom=1.15, translation_x=15, angle=3)
    combined.save(output_dir / "03_combined.png")
    print("   ✓ Zoom 15% + Pan 15px + Rotate 3°")
    
    print(f"\n✓ Test complete! Check {output_dir}/ for results")
    
    return image


def test_zoom_sequence():
    """
    Test 2: Create a zoom sequence (like Deforum does)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Zoom Sequence (10 frames)")
    print("=" * 70)
    
    # Generate starting image
    print("\n1. Generating start image...")
    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    
    current_image = generator.generate_from_text(
        prompt="mountain landscape, sunrise",
        seed=42
    )
    
    output_dir = Path("test_zoom_sequence")
    output_dir.mkdir(exist_ok=True)
    
    # Save frame 0
    current_image.save(output_dir / "frame_00.png")
    print("   ✓ Frame 0 generated")
    
    print("\n2. Creating zoom sequence...")
    zoom_per_frame = 0.02  # 2% zoom per frame
    
    for i in range(1, 10):
        # Apply zoom to previous frame
        zoom_factor = 1.0 + (zoom_per_frame * i)
        transformed = transform_image(current_image, zoom=zoom_factor)
        
        # Generate new frame from transformed image
        current_image = generator.generate_from_image(
            init_image=transformed,
            prompt="mountain landscape, sunrise",
            strength=0.5,  # Medium change
            seed=42 + i
        )
        
        current_image.save(output_dir / f"frame_{i:02d}.png")
        print(f"   ✓ Frame {i} (zoom: {zoom_factor:.2f})")
    
    print(f"\n✓ Sequence complete! Check {output_dir}/")
    print("  💡 Create video with: python frames_to_video.py --frames_dir test_zoom_sequence")


def test_beat_zoom():
    """
    Test 3: Demonstrate beat-reactive zoom
    
    Shows how zoom can pulse on beats (blueprint for music sync)
    """
    print("\n" + "=" * 70)
    print("TEST 3: Beat-Reactive Zoom (Blueprint)")
    print("=" * 70)
    
    print("""
Blueprint for combining transforms with music sync:

```python
from _src.music_sync import MusicSync
from _src.image_transform import transform_image
from _src.image_generator import ImageGenerator

# Setup
sync = MusicSync("song.mp3", fps=24)
sync.load()
generator = ImageGenerator()

current_image = generator.generate_from_text("landscape", seed=42)

# Animation loop
for frame_num in range(total_frames):
    is_beat = sync.is_beat_frame(frame_num)
    
    # Calculate zoom
    base_zoom = 1.0 + (frame_num * 0.01)  # Gradual zoom
    
    if is_beat:
        zoom = base_zoom + 0.05  # Extra zoom on beats!
    else:
        zoom = base_zoom
    
    # Transform previous frame
    transformed = transform_image(
        current_image,
        zoom=zoom,
        translation_x=2  # Slow pan right
    )
    
    # Generate new frame from transformed
    strength = sync.get_strength_for_frame(frame_num)
    current_image = generator.generate_from_image(
        transformed,
        prompt="landscape",
        strength=strength,
        seed=42 + frame_num
    )
```

Effects you'll see:
✓ Continuous zoom in
✓ Extra zoom pulse on beats
✓ Slow pan to the right
✓ Image changes synced with music
""")


def run_all_tests():
    """Run all transformation tests"""
    print("\n" + "=" * 70)
    print("🎬 IMAGE TRANSFORMATION TESTS")
    print("=" * 70)
    
    # Test 1: Basic transforms
    test_basic_transformations()
    
    # Test 2: Zoom sequence
    test_zoom_sequence()
    
    # Test 3: Beat zoom blueprint
    test_beat_zoom()
    
    print("\n" + "=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("  1. Check test_transforms/ for single transform examples")
    print("  2. Check test_zoom_sequence/ for animation sequence")
    print("  3. Combine with music_sync for beat-reactive effects")
    print()


if __name__ == "__main__":
    run_all_tests()
