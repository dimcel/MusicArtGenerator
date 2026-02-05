"""
Test Frame Interpolator and Video Animator with Interpolation

Tests:
1. Frame interpolator standalone
2. Video animator without interpolation (baseline)
3. Video animator with interpolation (fast mode)
4. Comparison of interpolation intervals
"""

import sys
from pathlib import Path
from PIL import Image

# Add _src to path
sys.path.insert(0, str(Path(__file__).parent))

from _src.frame_interpolator import FrameInterpolator, simple_interpolate
from _src._video_animator import VideoAnimator, VideoAnimationConfig, simple_animation


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
# Adjust these to control test video length
TOTAL_FRAMES = 200  # Number of frames for each test video
# TOTAL_FRAMES = 60   # Use this for faster testing (shorter videos)
# TOTAL_FRAMES = 400  # Use this for longer, more detailed videos
# =============================================================================


def test_interpolator_standalone():
    """Test: Frame interpolator with sample images"""
    print("\n" + "=" * 70)
    print("TEST 1: Frame Interpolator (Standalone)")
    print("=" * 70)
    
    print("\nGenerating 2 test keyframes...")
    
    from _src.image_generator import ImageGenerator, ImageGenerationConfig
    
    gen_config = ImageGenerationConfig(width=512, height=512)
    generator = ImageGenerator(gen_config)
    
    # Generate 2 different frames
    frame_0 = generator.generate_from_text(
        prompt="old man on bench, sunny day",
        seed=42
    )
    
    frame_4 = generator.generate_from_image(
        init_image=frame_0,
        prompt="old man on bench, sunny day",
        strength=0.7,  # Big change
        seed=46
    )
    
    print("   ✓ Generated keyframes")
    
    # Test interpolation
    print("\nInterpolating 3 frames between them...")
    interpolator = FrameInterpolator(method="blend")
    interpolated = interpolator.interpolate(frame_0, frame_4, num_frames=3)
    
    print(f"   ✓ Created {len(interpolated)} interpolated frames")
    
    # Save for visual inspection
    output_dir = Path("test_interpolator")
    output_dir.mkdir(exist_ok=True)
    
    frame_0.save(output_dir / "frame_0_keyframe.png")
    for i, frame in enumerate(interpolated, start=1):
        frame.save(output_dir / f"frame_{i}_interpolated.png")
    frame_4.save(output_dir / "frame_4_keyframe.png")
    
    print(f"\n✓ Test complete! Check {output_dir}/ for:")
    print("     frame_0_keyframe.png")
    print("     frame_1_interpolated.png (25% blend)")
    print("     frame_2_interpolated.png (50% blend)")
    print("     frame_3_interpolated.png (75% blend)")
    print("     frame_4_keyframe.png")


def test_video_no_interpolation():
    """Test: Video animator WITHOUT interpolation (baseline)"""
    print("\n" + "=" * 70)
    print("TEST 2: Video Animator - NO Interpolation (Baseline)")
    print("=" * 70)
    
    config = VideoAnimationConfig(
        prompt="old man sitting on a bench, peaceful autumn park",
        total_frames=TOTAL_FRAMES,  # Controlled by global variable
        fps=24,
        strength=0.6,
        use_interpolation=False,  # All frames generated
        output_dir="test_video_no_interp"
    )
    
    animator = VideoAnimator(config)
    video_path = animator.generate_and_save_video()
    
    print(f"\n✓ Baseline test complete: {video_path}")


def test_video_with_interpolation():
    """Test: Video animator WITH interpolation (fast mode)"""
    print("\n" + "=" * 70)
    print("TEST 3: Video Animator - WITH Interpolation (Fast)")
    print("=" * 70)
    
    config = VideoAnimationConfig(
        prompt="old man sitting on a bench, peaceful autumn park",
        total_frames=TOTAL_FRAMES,  # Controlled by global variable
        fps=24,
        strength=0.6,
        use_interpolation=True,   # Interpolation enabled
        keyframe_interval=4,      # Generate every 4th frame
        interpolation_method="optical_flow",  # Use optical flow method
        output_dir="test_video_with_interp"
    )
    
    animator = VideoAnimator(config)
    video_path = animator.generate_and_save_video()
    
    print(f"\n✓ Interpolation test complete: {video_path}")


def test_interpolation_intervals():
    """Test: Compare different interpolation intervals"""
    print("\n" + "=" * 70)
    print("TEST 4: Compare Interpolation Intervals")
    print("=" * 70)
    
    intervals = [2, 4, 8]
    
    for interval in intervals:
        print(f"\n--- Keyframe interval: {interval} ---")
        
        config = VideoAnimationConfig(
            prompt="old man on bench, golden sunset",
            total_frames=TOTAL_FRAMES,  # Controlled by global variable
            fps=24,
            strength=0.6,
            use_interpolation=True,
            keyframe_interval=interval,
            output_dir=f"test_video_interval_{interval}",
            seed=42  # Same seed for fair comparison
        )
        
        animator = VideoAnimator(config)
        video_path = animator.generate_and_save_video()
        
        print(f"   ✓ Created: {video_path}")
    
    print("\n✓ All interval tests complete!")
    print("\nCompare the videos:")
    print(f"  - interval=2 → Generate {TOTAL_FRAMES // 2} keyframes (2x speedup)")
    print(f"  - interval=4 → Generate {TOTAL_FRAMES // 4} keyframes (4x speedup)")
    print(f"  - interval=8 → Generate {TOTAL_FRAMES // 8} keyframes (8x speedup)")
    print("\nNote: Higher intervals = faster but more ghosting artifacts")


def test_simple_helper():
    """Test: Simple helper function with interpolation"""
    print("\n" + "=" * 70)
    print("TEST 5: Simple Helper Function")
    print("=" * 70)
    
    video_path = simple_animation(
        prompt="old man on bench, thinking deeply",
        frames=TOTAL_FRAMES,  # Controlled by global variable
        strength=0.6,
        use_interpolation=True,
        keyframe_interval=4,
        output_dir="test_simple_animation"
    )
    
    print(f"\n✓ Simple animation complete: {video_path}")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 FRAME INTERPOLATION TESTS")
    print("=" * 70)
    print("\nThis tests the new interpolation system:")
    print("  1. Frame interpolator (blend method)")
    print("  2. Video animator with/without interpolation")
    print("  3. Different keyframe intervals")
    print("\nNote: Generation will take several minutes.")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                       help="Run specific test (1-5), or 0 for all")
    args = parser.parse_args()
    
    try:
        if args.test == 0 or args.test == 1:
            test_interpolator_standalone()
        
        if args.test == 0 or args.test == 2:
            test_video_no_interpolation()
        
        if args.test == 0 or args.test == 3:
            test_video_with_interpolation()
        
        if args.test == 0 or args.test == 4:
            test_interpolation_intervals()
        
        if args.test == 0 or args.test == 5:
            test_simple_helper()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nGenerated outputs:")
        print("  test_interpolator/ - Individual interpolated frames")
        print("  test_video_no_interp/ - Baseline (all frames generated)")
        print("  test_video_with_interp/ - With interpolation")
        print("  test_video_interval_*/ - Different intervals comparison")
        print("  test_simple_animation/ - Simple helper test")
        print("\nCompare the videos to see interpolation quality!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
