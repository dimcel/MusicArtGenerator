"""
Test script for Image Generator

Simple test to verify the refactored image generation works.
"""

import sys
from pathlib import Path

# Add _src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_generator import ImageGenerator, ImageGenerationConfig


def test_text_to_image():
    """Test: Generate image from text prompt"""
    print("\n" + "=" * 70)
    print("TEST 1: Text-to-Image (First Frame)")
    print("=" * 70)
    
    # Create generator
    config = ImageGenerationConfig(
        width=512,
        height=512,
        num_inference_steps=20  # Fast for testing
    )
    generator = ImageGenerator(config)
    
    # Generate image
    print("\n📝 Prompt: 'old man sitting on a bench, thinking, autumn park'")
    image = generator.generate_from_text(
        prompt="old man sitting on a bench, thinking, peaceful autumn park, photorealistic",
        seed=42
    )
    
    # Save result
    output_path = "test_output/test_txt2img.png"
    Path("test_output").mkdir(exist_ok=True)
    image.save(output_path)
    
    print(f"\n✓ Generated image saved: {output_path}")
    print(f"  Size: {image.size}")
    
    return image


def test_image_to_image(init_image):
    """Test: Generate image from existing image"""
    print("\n" + "=" * 70)
    print("TEST 2: Image-to-Image (Subsequent Frame)")
    print("=" * 70)
    
    # Create generator (reuses if already loaded)
    config = ImageGenerationConfig(
        num_inference_steps=20  # Fast for testing
    )
    generator = ImageGenerator(config)
    
    # Generate next frame
    print("\n📝 Prompt: 'old man sitting on a bench, golden hour light'")
    print("   Strength: 0.6 (moderate change)")
    
    image = generator.generate_from_image(
        init_image=init_image,
        prompt="old man sitting on a bench, warm golden hour light, peaceful atmosphere, photorealistic",
        strength=0.6,
        seed=42
    )
    
    # Save result
    output_path = "test_output/test_img2img.png"
    image.save(output_path)
    
    print(f"\n✓ Generated image saved: {output_path}")
    print(f"  Size: {image.size}")
    
    return image


def test_strength_variation(init_image):
    """Test: Different strength values"""
    print("\n" + "=" * 70)
    print("TEST 3: Strength Variation")
    print("=" * 70)
    
    config = ImageGenerationConfig(num_inference_steps=15)
    generator = ImageGenerator(config)
    
    strengths = [0.3, 0.5, 0.7]
    prompt = "old man sitting on a bench, sunset, dramatic lighting, photorealistic"
    
    for strength in strengths:
        print(f"\n   Generating with strength={strength}...")
        
        image = generator.generate_from_image(
            init_image=init_image,
            prompt=prompt,
            strength=strength,
            seed=42
        )
        
        output_path = f"test_output/test_strength_{strength}.png"
        image.save(output_path)
        print(f"   ✓ Saved: {output_path}")
    
    print("\n✓ All strength variations complete")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 IMAGE GENERATOR TESTS")
    print("=" * 70)
    print("\nThis tests the refactored image generation module.")
    print("It will generate a few test images (takes ~2-3 minutes).")
    
    try:
        # Test 1: Text-to-Image
        first_image = test_text_to_image()
        
        # Test 2: Image-to-Image
        second_image = test_image_to_image(first_image)
        
        # Test 3: Strength variations
        test_strength_variation(first_image)
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nCheck the test_output/ folder for generated images.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
