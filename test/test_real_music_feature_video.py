"""Compatibility launcher for the music video app.

Legacy entrypoint retained for existing workflows:
- python test/test_real_music_feature_video.py --audio ...

Preferred package entrypoint:
- music-art-generator --audio ...
"""

from music_art_generator.music_video_app.cli import main


def run(*args, **kwargs):
    from music_art_generator.music_video_app.runner import run as _run

    return _run(*args, **kwargs)

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
