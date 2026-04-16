"""
Refactored Animation Generator (src)

Clean, modular implementation following best coding principles.

Modules:
- image_generator: Generate images with Stable Diffusion
- (more modules coming as we refactor step by step)
"""

__all__ = [
    'ImageGenerator',
    'ImageGenerationConfig',
]

__version__ = '2.0.0-refactor'


def __getattr__(name):
    if name in ("ImageGenerator", "ImageGenerationConfig"):
        from src.image_generator import ImageGenerator, ImageGenerationConfig

        mapping = {
            "ImageGenerator": ImageGenerator,
            "ImageGenerationConfig": ImageGenerationConfig,
        }
        return mapping[name]
    raise AttributeError(f"module 'src' has no attribute '{name}'")
