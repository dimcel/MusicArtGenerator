"""
Audio Feature Extractor (Simple)

Extracts a few music features and maps them to video frames:
- beat frames
- beat pulse curve (0..1)
- rms energy curve (0..1)
- onset curve (0..1)
- spectral centroid / "brightness" curve (0..1)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class AudioFrameFeatures:
    fps: int
    total_frames: int
    duration_seconds: float
    bpm: float
    beat_frames: List[int]
    beat_pulse: np.ndarray
    energy: np.ndarray
    onset: np.ndarray
    brightness: np.ndarray


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min()) if len(arr) else 0.0
    hi = float(arr.max()) if len(arr) else 1.0
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _smooth(arr: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return arr
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(arr, kernel, mode="same")


def _pulse_from_events(total_frames: int, event_frames: List[int], decay_frames: int = 2) -> np.ndarray:
    out = np.zeros(total_frames, dtype=np.float32)
    if decay_frames <= 0:
        for f in event_frames:
            if 0 <= f < total_frames:
                out[f] = 1.0
        return out

    for i in range(total_frames):
        best = 0.0
        for ev in event_frames:
            d = abs(i - ev)
            if d <= decay_frames:
                v = 1.0 - (d / float(decay_frames))
                if v > best:
                    best = v
        out[i] = best
    return out


class AudioFeatureExtractor:
    """
    Extract frame-aligned audio features for animation control.
    """

    def __init__(self, audio_path: str, fps: int = 24):
        self.audio_path = audio_path
        self.fps = fps
        self.y: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.duration_seconds: Optional[float] = None

    def load(self) -> "AudioFeatureExtractor":
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio feature extraction.\n"
                "Install: pip install librosa"
            )

        if not Path(self.audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        self.y, self.sr = librosa.load(self.audio_path, sr=22050)
        self.duration_seconds = len(self.y) / float(self.sr)
        return self

    def _ensure_loaded(self):
        if self.y is None or self.sr is None or self.duration_seconds is None:
            raise RuntimeError("Audio not loaded. Call load() first.")

    def extract(self, total_frames: Optional[int] = None) -> AudioFrameFeatures:
        """
        Extract and frame-align core features.

        If total_frames is None, uses full audio duration at configured fps.
        """
        self._ensure_loaded()
        import librosa

        if total_frames is None:
            total_frames = max(1, int(self.duration_seconds * self.fps))

        # Beat tracking.
        tempo, beat_frames_lib = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames_lib, sr=self.sr)
        bpm = float(np.atleast_1d(tempo)[0])
        beat_frames = sorted(set(int(t * self.fps) for t in beat_times if int(t * self.fps) < total_frames))
        beat_pulse = _pulse_from_events(total_frames, beat_frames, decay_frames=2)

        # Time series from librosa frame-rate to video frame-rate.
        rms = librosa.feature.rms(y=self.y)[0]
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]

        rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr)
        onset_t = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr)
        cent_t = librosa.frames_to_time(np.arange(len(centroid)), sr=self.sr)
        frame_t = np.arange(total_frames, dtype=np.float32) / float(self.fps)

        energy = np.interp(frame_t, rms_t, _normalize(rms)).astype(np.float32)
        onset = np.interp(frame_t, onset_t, _normalize(onset_env)).astype(np.float32)
        brightness = np.interp(frame_t, cent_t, _normalize(centroid)).astype(np.float32)

        # Small smoothing for stability.
        energy = _smooth(energy, win=5).astype(np.float32)
        onset = _smooth(onset, win=3).astype(np.float32)
        brightness = _smooth(brightness, win=7).astype(np.float32)

        return AudioFrameFeatures(
            fps=self.fps,
            total_frames=total_frames,
            duration_seconds=float(self.duration_seconds),
            bpm=bpm,
            beat_frames=beat_frames,
            beat_pulse=beat_pulse,
            energy=energy,
            onset=onset,
            brightness=brightness,
        )

