"""Compatibility wrapper for the packaged frame-to-video utility."""

from music_art_generator.frames_to_video import frames_to_video, list_frames, main

__all__ = ["frames_to_video", "list_frames", "main"]


if __name__ == "__main__":
    main()
