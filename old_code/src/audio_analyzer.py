"""
Audio Analyzer for Animation Engine
Extracts beats, amplitude, and onsets from audio files for music synchronization

Dependencies:
    pip install librosa numpy --break-system-packages
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    duration: float                    # Total duration in seconds
    bpm: float                         # Detected tempo
    beat_times: np.ndarray            # Timestamps of beats (seconds)
    onset_times: np.ndarray           # Timestamps of transients/hits (seconds)
    amplitude_times: np.ndarray       # Timestamps for amplitude envelope
    amplitude_values: np.ndarray      # Amplitude values (0-1 normalized)


class AudioAnalyzer:
    """
    Extracts musical features from audio files for animation synchronization.
    
    Usage:
        analyzer = AudioAnalyzer("song.mp3")
        analyzer.load()
        
        # Get features
        bpm = analyzer.get_bpm()
        beats = analyzer.get_beat_times()
        amplitude = analyzer.get_amplitude_envelope()
    """
    
    def __init__(self, audio_path: str):
        """
        Initialize analyzer with path to audio file.
        
        Args:
            audio_path: Path to audio file (mp3, wav, flac, etc.)
        """
        self.audio_path = audio_path
        self.y: Optional[np.ndarray] = None      # Audio samples
        self.sr: Optional[int] = None            # Sample rate
        self.duration: Optional[float] = None    # Duration in seconds
        
        # Cached features (computed on demand)
        self._bpm: Optional[float] = None
        self._beat_times: Optional[np.ndarray] = None
        self._onset_times: Optional[np.ndarray] = None
        self._amplitude_times: Optional[np.ndarray] = None
        self._amplitude_values: Optional[np.ndarray] = None
    
    def load(self) -> 'AudioAnalyzer':
        """
        Load audio file into memory.
        
        Returns:
            self (for chaining)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required. Install with:\n"
                "pip install librosa --break-system-packages"
            )
        
        print(f"Loading audio: {self.audio_path}")
        
        # Load audio (librosa automatically resamples to sr=22050 by default)
        self.y, self.sr = librosa.load(self.audio_path, sr=22050)
        self.duration = len(self.y) / self.sr
        
        print(f"  Duration: {self.duration:.2f}s")
        print(f"  Sample rate: {self.sr} Hz")
        print(f"  Samples: {len(self.y):,}")
        
        return self
    
    def _ensure_loaded(self):
        """Check that audio is loaded"""
        if self.y is None:
            raise RuntimeError("Audio not loaded. Call load() first.")
    
    # =========================================================================
    # TEMPO & BEATS
    # =========================================================================
    
    def get_bpm(self) -> float:
        """
        Detect tempo (beats per minute).
        
        Returns:
            Estimated BPM
        """
        self._ensure_loaded()
        
        if self._bpm is None:
            import librosa
            
            # Use librosa's beat tracker
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            
            # tempo can be an array in newer versions
            self._bpm = float(np.atleast_1d(tempo)[0])
            print(f"  Detected BPM: {self._bpm:.1f}")
        
        return self._bpm
    
    def get_beat_times(self) -> np.ndarray:
        """
        Get timestamps of detected beats.
        
        Returns:
            Array of beat timestamps in seconds
        """
        self._ensure_loaded()
        
        if self._beat_times is None:
            import librosa
            
            # Get beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
            
            # Convert frames to time
            self._beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
            self._bpm = float(np.atleast_1d(tempo)[0])
            
            print(f"  Detected {len(self._beat_times)} beats")
        
        return self._beat_times
    
    # =========================================================================
    # AMPLITUDE / ENERGY
    # =========================================================================
    
    def get_amplitude_envelope(self, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitude envelope from audio.
        
        Args:
            hop_length: Number of samples between frames (lower = more resolution)
        
        Returns:
            Tuple of (times, amplitudes) where amplitudes are normalized 0-1
        """
        self._ensure_loaded()
        
        if self._amplitude_values is None:
            import librosa
            
            # Compute RMS energy
            rms = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]
            
            # Get corresponding timestamps
            times = librosa.frames_to_time(
                np.arange(len(rms)), 
                sr=self.sr, 
                hop_length=hop_length
            )
            
            # Normalize to 0-1
            rms_normalized = rms / (rms.max() + 1e-8)
            
            self._amplitude_times = times
            self._amplitude_values = rms_normalized
            
            print(f"  Extracted {len(rms)} amplitude samples")
        
        return self._amplitude_times, self._amplitude_values
    
    # =========================================================================
    # ONSETS (Transients / Hits)
    # =========================================================================
    
    def get_onsets(self) -> np.ndarray:
        """
        Detect onset times (sudden changes in energy - drums, attacks, etc.)
        
        Returns:
            Array of onset timestamps in seconds
        """
        self._ensure_loaded()
        
        if self._onset_times is None:
            import librosa
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=self.y, sr=self.sr)
            self._onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            
            print(f"  Detected {len(self._onset_times)} onsets")
        
        return self._onset_times
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_amplitude_at_time(self, time: float) -> float:
        """
        Get interpolated amplitude value at specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Amplitude value (0-1)
        """
        times, values = self.get_amplitude_envelope()
        
        # Find nearest index
        idx = np.searchsorted(times, time)
        idx = np.clip(idx, 0, len(values) - 1)
        
        return float(values[idx])
    
    def is_beat_near(self, time: float, tolerance: float = 0.05) -> bool:
        """
        Check if a time is near a beat.
        
        Args:
            time: Time in seconds
            tolerance: How close to beat (in seconds) counts as "on beat"
            
        Returns:
            True if time is within tolerance of a beat
        """
        beat_times = self.get_beat_times()
        
        # Check if any beat is within tolerance
        distances = np.abs(beat_times - time)
        return bool(np.any(distances <= tolerance))
    
    def get_all_features(self) -> AudioFeatures:
        """
        Extract all features at once.
        
        Returns:
            AudioFeatures dataclass with all extracted data
        """
        self._ensure_loaded()
        
        amp_times, amp_values = self.get_amplitude_envelope()
        
        return AudioFeatures(
            duration=self.duration,
            bpm=self.get_bpm(),
            beat_times=self.get_beat_times(),
            onset_times=self.get_onsets(),
            amplitude_times=amp_times,
            amplitude_values=amp_values
        )
    
    def summary(self) -> str:
        """Get a summary of the audio analysis"""
        self._ensure_loaded()
        
        features = self.get_all_features()
        
        return f"""
Audio Analysis Summary
======================
File: {self.audio_path}
Duration: {features.duration:.2f} seconds
BPM: {features.bpm:.1f}
Beats detected: {len(features.beat_times)}
Onsets detected: {len(features.onset_times)}
Amplitude samples: {len(features.amplitude_values)}
"""


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

def example_usage():
    """
    Example of how to use the AudioAnalyzer
    """
    print("=" * 60)
    print("AUDIO ANALYZER - Example Usage")
    print("=" * 60)
    
    # Create a simple test tone if no audio file available
    print("\nCreating test audio (sine wave with beats)...")
    
    sr = 22050
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple beat pattern (kick every 0.5 seconds = 120 BPM)
    beat_times = np.arange(0, duration, 0.5)
    
    # Generate audio: sine wave + impulses at beats
    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # Base tone
    
    for beat_time in beat_times:
        # Add a short impulse at each beat
        beat_idx = int(beat_time * sr)
        impulse_len = int(0.05 * sr)  # 50ms impulse
        if beat_idx + impulse_len < len(audio):
            decay = np.exp(-np.linspace(0, 5, impulse_len))
            audio[beat_idx:beat_idx + impulse_len] += decay * 0.7
    
    # Save test audio
    test_file = "/tmp/test_audio.wav"
    try:
        import soundfile as sf
        sf.write(test_file, audio, sr)
        print(f"  Saved test audio to {test_file}")
    except ImportError:
        print("  (soundfile not available, skipping file save)")
        # Create analyzer with fake data for demo
        print("\n  Demonstrating with synthetic data...")
        
        # Create mock analyzer
        analyzer = AudioAnalyzer("fake_path.mp3")
        analyzer.y = audio
        analyzer.sr = sr
        analyzer.duration = duration
        
        print("\n" + analyzer.summary())
        
        print("\nBeat times (first 10):")
        beats = analyzer.get_beat_times()
        print(f"  {beats[:10]}")
        
        print("\nAmplitude at various times:")
        for test_time in [0.0, 0.5, 1.0, 1.5, 2.0]:
            amp = analyzer.get_amplitude_at_time(test_time)
            is_beat = analyzer.is_beat_near(test_time)
            print(f"  t={test_time:.1f}s: amplitude={amp:.3f}, on_beat={is_beat}")
        
        return analyzer
    
    # If we could save the file, load it properly
    analyzer = AudioAnalyzer(test_file)
    analyzer.load()
    
    print("\n" + analyzer.summary())
    
    print("\nBeat times (first 10):")
    beats = analyzer.get_beat_times()
    print(f"  {beats[:10]}")
    
    print("\nAmplitude at various times:")
    for test_time in [0.0, 0.5, 1.0, 1.5, 2.0]:
        amp = analyzer.get_amplitude_at_time(test_time)
        is_beat = analyzer.is_beat_near(test_time)
        print(f"  t={test_time:.1f}s: amplitude={amp:.3f}, on_beat={is_beat}")
    
    return analyzer


if __name__ == "__main__":
    example_usage()