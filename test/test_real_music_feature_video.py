"""Compatibility launcher for the music video app.

Legacy entrypoint retained for existing workflows:
- python test/test_real_music_feature_video.py --audio ...

Preferred package entrypoint:
- python -m src.music_video_app --audio ...
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))

from src.music_video_app.cli import main


def run(*args, **kwargs):
    from src.music_video_app.runner import run as _run

    return _run(*args, **kwargs)

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
