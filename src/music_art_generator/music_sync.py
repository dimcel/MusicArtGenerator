"""
Music Synchronization Module
Simple beat detection for syncing video generation with music

Inspired by sd-parseq's approach to audio-driven parameter scheduling.
This module detects beats and creates a schedule for when to generate frames.

Usage:
    sync = MusicSync("song.mp3", fps=24)
    sync.load()
    
    # Check if frame should be generated on a beat
    if sync.is_beat_frame(frame_num):
        strength = sync.get_strength_for_frame(frame_num)  # Higher on beats
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BeatSchedule:
    """
    Schedule defining when beats occur in the video.
    
    Attributes:
        beat_times: List of beat timestamps in seconds
        beat_frames: List of frame numbers where beats occur
        fps: Frames per second
        bpm: Detected beats per minute
    """
    beat_times: np.ndarray
    beat_frames: List[int]
    fps: int
    bpm: float
    
    def __len__(self):
        return len(self.beat_frames)


class MusicSync:
    """
    Simple music synchronization based on beat detection.
    
    Similar to sd-parseq's approach:
    - Detects beats from audio
    - Creates keyframe schedule at beat positions
    - Modulates generation parameters (strength, seed) based on beats
    
    Example:
        sync = MusicSync("song.mp3", fps=24)
        sync.load()
        
        # During animation generation
        for frame in range(total_frames):
            if sync.is_beat_frame(frame):
                print(f"Frame {frame} is on a beat!")
                strength = 0.85  # More change
            else:
                strength = 0.4   # Smoother
    """
    
    def __init__(
        self, 
        audio_path: str, 
        fps: int = 24,
        beat_strength: float = 0.85,
        normal_strength: float = 0.4
    ):
        """
        Initialize music sync.
        
        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            fps: Video frames per second
            beat_strength: Generation strength on beat frames (0-1)
            normal_strength: Generation strength between beats (0-1)
        """
        self.audio_path = audio_path
        self.fps = fps
        self.beat_strength = beat_strength
        self.normal_strength = normal_strength
        
        # Will be populated after load()
        self.y: Optional[np.ndarray] = None      # Audio samples
        self.sr: Optional[int] = None            # Sample rate
        self.duration: Optional[float] = None    # Duration in seconds
        self.schedule: Optional[BeatSchedule] = None
    
    def load(self) -> 'MusicSync':
        """
        Load audio file and detect beats.
        
        Returns:
            self (for chaining)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for music sync. Install with:\n"
                "  pip install librosa\n"
                "Or on macOS:\n"
                "  pip install librosa --break-system-packages"
            )
        
        print(f"🎵 Loading audio: {self.audio_path}")
        
        # Load audio
        self.y, self.sr = librosa.load(self.audio_path, sr=22050)
        self.duration = len(self.y) / self.sr
        
        print(f"   Duration: {self.duration:.2f}s")
        print(f"   Sample rate: {self.sr} Hz")
        
        # Detect beats
        print(f"   Detecting beats...")
        tempo, beat_frames_librosa = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames_librosa, sr=self.sr)
        
        # Convert tempo to float (can be array in newer versions)
        bpm = float(np.atleast_1d(tempo)[0])
        
        # Convert beat times to video frame numbers
        beat_video_frames = [int(t * self.fps) for t in beat_times]
        
        # Create schedule
        self.schedule = BeatSchedule(
            beat_times=beat_times,
            beat_frames=beat_video_frames,
            fps=self.fps,
            bpm=bpm
        )
        
        print(f"   ✓ BPM: {bpm:.1f}")
        print(f"   ✓ Detected {len(beat_times)} beats")
        print(f"   ✓ Video frames with beats: {len(beat_video_frames)}")
        
        return self
    
    def _ensure_loaded(self):
        """Check that audio is loaded"""
        if self.schedule is None:
            raise RuntimeError("Audio not loaded. Call load() first.")
    
    # =========================================================================
    # BEAT QUERIES
    # =========================================================================
    
    def is_beat_frame(self, frame_num: int) -> bool:
        """
        Check if a frame number is on a beat.
        
        Args:
            frame_num: Frame number to check
            
        Returns:
            True if frame is on a beat
        """
        self._ensure_loaded()
        return frame_num in self.schedule.beat_frames
    
    def get_nearest_beat_frame(self, frame_num: int) -> int:
        """
        Get the nearest beat frame to a given frame.
        
        Args:
            frame_num: Frame number to check
            
        Returns:
            Nearest beat frame number
        """
        self._ensure_loaded()
        
        if not self.schedule.beat_frames:
            return 0
        
        # Find nearest beat
        distances = [abs(bf - frame_num) for bf in self.schedule.beat_frames]
        nearest_idx = np.argmin(distances)
        return self.schedule.beat_frames[nearest_idx]
    
    def get_beat_frames(self) -> List[int]:
        """
        Get all beat frame numbers.
        
        Returns:
            List of frame numbers where beats occur
        """
        self._ensure_loaded()
        return self.schedule.beat_frames
    
    # =========================================================================
    # PARAMETER MODULATION (sd-parseq style)
    # =========================================================================
    
    def get_strength_for_frame(self, frame_num: int) -> float:
        """
        Get generation strength for a frame (higher on beats).
        
        This is the core of Strategy 1: modulate the img2img strength
        parameter based on whether we're on a beat or not.
        
        Args:
            frame_num: Frame number
            
        Returns:
            Strength value (0-1)
        """
        self._ensure_loaded()
        
        if self.is_beat_frame(frame_num):
            return self.beat_strength
        else:
            return self.normal_strength
    
    def get_seed_offset_for_frame(self, frame_num: int, base_seed: int = 42) -> int:
        """
        Get seed value for a frame (jumps on beats).
        
        Args:
            frame_num: Frame number
            base_seed: Base seed value
            
        Returns:
            Seed offset to add to base seed
        """
        self._ensure_loaded()
        
        if self.is_beat_frame(frame_num):
            # On beats, jump to a new seed for more variation
            beat_index = self.schedule.beat_frames.index(frame_num)
            return beat_index * 100  # Large jump
        else:
            # Between beats, use frame number for smooth progression
            return frame_num
    
    # =========================================================================
    # KEYFRAME SCHEDULE (sd-parseq style)
    # =========================================================================
    
    def get_keyframe_schedule(self, max_frames: Optional[int] = None) -> List[int]:
        """
        Get list of frames that should be generated (not interpolated).
        
        This returns beat frames that can be used as keyframes,
        with interpolation filling the gaps between them.
        
        Args:
            max_frames: Maximum frame number (optional)
            
        Returns:
            List of frame numbers to generate as keyframes
        """
        self._ensure_loaded()
        
        # Start with all beat frames
        keyframes = [0] + self.schedule.beat_frames  # Include frame 0
        
        # Remove duplicates and sort
        keyframes = sorted(set(keyframes))
        
        # Filter by max_frames if provided
        if max_frames is not None:
            keyframes = [f for f in keyframes if f < max_frames]
        
        return keyframes
    
    # =========================================================================
    # SUMMARY & INFO
    # =========================================================================
    
    def get_info(self) -> dict:
        """
        Get summary information about the music sync.
        
        Returns:
            Dictionary with audio and beat information
        """
        self._ensure_loaded()
        
        return {
            'audio_path': self.audio_path,
            'duration_seconds': self.duration,
            'fps': self.fps,
            'bpm': self.schedule.bpm,
            'num_beats': len(self.schedule.beat_frames),
            'beat_frames': self.schedule.beat_frames[:10],  # First 10
            'beat_strength': self.beat_strength,
            'normal_strength': self.normal_strength,
        }
    
    def print_schedule(self, max_display: int = 20):
        """
        Print beat schedule in readable format.
        
        Args:
            max_display: Maximum number of beats to display
        """
        self._ensure_loaded()
        
        print(f"\n{'='*60}")
        print(f"BEAT SCHEDULE")
        print(f"{'='*60}")
        print(f"Audio: {self.audio_path}")
        print(f"BPM: {self.schedule.bpm:.1f}")
        print(f"FPS: {self.fps}")
        print(f"Total beats: {len(self.schedule.beat_frames)}")
        print(f"\nFirst {max_display} beats:")
        print(f"{'Beat':<6} {'Time':<10} {'Frame':<10}")
        print(f"{'-'*30}")
        
        for i in range(min(max_display, len(self.schedule.beat_times))):
            beat_time = self.schedule.beat_times[i]
            beat_frame = self.schedule.beat_frames[i]
            print(f"{i+1:<6} {beat_time:>6.2f}s    Frame {beat_frame:<5}")
        
        if len(self.schedule.beat_frames) > max_display:
            print(f"... and {len(self.schedule.beat_frames) - max_display} more")
        
        print(f"{'='*60}\n")


# =============================================================================
# SIMPLE HELPER FUNCTION
# =============================================================================

def simple_beat_detection(audio_path: str, fps: int = 24) -> BeatSchedule:
    """
    Quick helper to detect beats without class setup.
    
    Args:
        audio_path: Path to audio file
        fps: Video frames per second
        
    Returns:
        BeatSchedule with beat information
        
    Example:
        schedule = simple_beat_detection("song.mp3", fps=24)
        print(f"Beats at frames: {schedule.beat_frames}")
    """
    sync = MusicSync(audio_path, fps=fps)
    sync.load()
    return sync.schedule
