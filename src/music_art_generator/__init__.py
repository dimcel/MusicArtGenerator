"""MusicArtGenerator public package."""

__all__ = [
    "ImageGenerator",
    "ImageGenerationConfig",
]

__version__ = "1.0.0"


def __getattr__(name):
    if name in ("ImageGenerator", "ImageGenerationConfig"):
        from music_art_generator.image_generator import ImageGenerationConfig, ImageGenerator

        mapping = {
            "ImageGenerator": ImageGenerator,
            "ImageGenerationConfig": ImageGenerationConfig,
        }
        return mapping[name]
    raise AttributeError(f"module 'music_art_generator' has no attribute '{name}'")
