"""
Refactored Animation Generator (_src)

Clean, modular implementation following best coding principles.

Modules:
- image_generator: Generate images with Stable Diffusion
- (more modules coming as we refactor step by step)
"""

from _src.image_generator import ImageGenerator, ImageGenerationConfig

__all__ = [
    'ImageGenerator',
    'ImageGenerationConfig',
]

__version__ = '2.0.0-refactor'
