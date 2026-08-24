"""
Test Video Animator

Tests the Deforum-style animation loop:
Frame 0: prompt + noise → diffusion → image
Frame N: previous_image + noise → diffusion → new_image
"""

from music_art_generator.video_animator import VideoAnimator, VideoAnimationConfig, simple_animation


def test_short_animation():
    """Test: Generate short animation (30 frames = 1.25s)"""
    print("\n" + "=" * 70)
    print("TEST 1: Short Animation (30 frames)")
    print("=" * 70)
    
    config = VideoAnimationConfig(
        prompt="old man sitting on a bench, thinking, peaceful autumn park, photorealistic",
        negative_prompt="blurry, bad quality, distorted, ugly",
        total_frames=30,  # Short for testing
        fps=24,
        strength=0.6,  # Moderate change per frame
        output_dir="test_animation_short",
        width=512,
        height=512,
        seed=42
    )
    
    animator = VideoAnimator(config)
    video_path = animator.generate_and_save_video()
    
    print(f"\n✓ Test complete!")
    print(f"  Check: {video_path}")
    
    return video_path


def test_strength_comparison():
    """Test: Compare different strength values"""
    print("\n" + "=" * 70)
    print("TEST 2: Strength Comparison")
    print("=" * 70)
    print("\nGenerating 3 short videos with different strengths...")
    
    prompt = "old man sitting on a bench, golden sunset, photorealistic"
    strengths = [0.3, 0.5, 0.7]
    
    for strength in strengths:
        print(f"\n--- Strength: {strength} ---")
        
        config = VideoAnimationConfig(
            prompt=prompt,
            total_frames=20,  # Very short
            fps=24,
            strength=strength,
            output_dir=f"test_animation_strength_{strength}",
            seed=42
        )
        
        animator = VideoAnimator(config)
        video_path = animator.generate_and_save_video()
        
        print(f"   ✓ Created: {video_path}")
    
    print("\n✓ All strength tests complete!")
    print("\nCompare the videos:")
    print("  - strength=0.3 → subtle changes (more stable)")
    print("  - strength=0.5 → moderate changes")
    print("  - strength=0.7 → dramatic changes (less stable)")


def test_simple_helper():
    """Test: Use the simple helper function"""
    print("\n" + "=" * 70)
    print("TEST 3: Simple Helper Function")
    print("=" * 70)
    
    video_path = simple_animation(
        prompt="old man on bench, peaceful atmosphere, autumn colors",
        frames=25,
        strength=0.6,
        output_dir="test_animation_simple"
    )
    
    print(f"\n✓ Simple animation complete: {video_path}")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 VIDEO ANIMATOR TESTS")
    print("=" * 70)
    print("\nThis tests the Deforum-style animation generation.")
    print("Each frame is generated from the previous frame.")
    print("\nNote: This will take several minutes depending on GPU speed.")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0,
                       help="Run specific test (1, 2, or 3), or 0 for all")
    args = parser.parse_args()
    
    try:
        if args.test == 0 or args.test == 1:
            test_short_animation()
        
        if args.test == 0 or args.test == 2:
            test_strength_comparison()
        
        if args.test == 0 or args.test == 3:
            test_simple_helper()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nCheck the test_animation_*/ folders for:")
        print("  - frame_*.png (individual frames)")
        print("  - animation.mp4 (final video)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
