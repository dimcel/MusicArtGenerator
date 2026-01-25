"""
Music Schedule for Animation Engine
Converts audio features (from AudioAnalyzer) to per-frame parameter modifiers

This bridges the gap between:
  - Audio time (seconds, beats) 
  - Animation time (frames)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from audio_analyzer import AudioAnalyzer


@dataclass
class MusicSyncConfig:
    """
    Configuration for how audio affects animation parameters.
    
    Example:
        config = MusicSyncConfig(
            amplitude_to_strength=(0.5, 0.8),  # strength ranges 0.5-0.8 based on amplitude
            beat_zoom_boost=0.1,               # zoom increases by 0.1 on beats
        )
    """
    # Amplitude mapping: (min_value, max_value)
    # amplitude 0 → min_value, amplitude 1 → max_value
    amplitude_to_strength: Optional[Tuple[float, float]] = None
    amplitude_to_noise: Optional[Tuple[float, float]] = None
    amplitude_to_guidance: Optional[Tuple[float, float]] = None
    
    # Beat effects: value to ADD on beat frames
    beat_strength_boost: float = 0.0
    beat_zoom_boost: float = 0.0
    beat_noise_boost: float = 0.0
    
    # Onset effects (transients/hits)
    onset_strength_boost: float = 0.0
    onset_zoom_boost: float = 0.0
    
    # Smoothing (frames to smooth over, 0 = no smoothing)
    amplitude_smoothing: int = 0
    
    # Beat detection tolerance (seconds)
    beat_tolerance: float = 0.05
    onset_tolerance: float = 0.03


class MusicSchedule:
    """
    Converts audio analysis to per-frame parameter modifiers.
    
    Usage:
        analyzer = AudioAnalyzer("song.mp3")
        analyzer.load()
        
        music = MusicSchedule(analyzer, fps=24, total_frames=120)
        
        # Get amplitude at frame
        amp = music.get_amplitude_at_frame(30)
        
        # Check if frame is on a beat
        is_beat = music.is_beat_frame(30)
        
        # Generate full modifier schedule
        modifiers = music.generate_modifiers(config)
    """
    
    def __init__(self, analyzer: AudioAnalyzer, fps: int, total_frames: int):
        """
        Initialize MusicSchedule.
        
        Args:
            analyzer: AudioAnalyzer with loaded audio
            fps: Frames per second of animation
            total_frames: Total number of frames in animation
        """
        self.analyzer = analyzer
        self.fps = fps
        self.total_frames = total_frames
        
        # Pre-compute frame-based data for speed
        self._amplitude_per_frame: Optional[np.ndarray] = None
        self._beat_frames: Optional[set] = None
        self._onset_frames: Optional[set] = None
        
        # Validate
        if analyzer.duration is None:
            raise RuntimeError("AudioAnalyzer must be loaded first. Call analyzer.load()")
        
        animation_duration = total_frames / fps
        if animation_duration > analyzer.duration:
            print(f"  Warning: Animation ({animation_duration:.1f}s) is longer than audio ({analyzer.duration:.1f}s)")
    
    # =========================================================================
    # TIME <-> FRAME CONVERSION
    # =========================================================================
    
    def time_to_frame(self, seconds: float) -> int:
        """Convert time in seconds to frame number"""
        frame = int(seconds * self.fps)
        return min(frame, self.total_frames - 1)
    
    def frame_to_time(self, frame: int) -> float:
        """Convert frame number to time in seconds"""
        return frame / self.fps
    
    def beat_to_frame(self, beat_number: int) -> int:
        """Convert beat number to frame (requires BPM)"""
        bpm = self.analyzer.get_bpm()
        seconds_per_beat = 60.0 / bpm
        return self.time_to_frame(beat_number * seconds_per_beat)
    
    # =========================================================================
    # AMPLITUDE
    # =========================================================================
    
    def _compute_amplitude_per_frame(self) -> np.ndarray:
        """Pre-compute amplitude values for every frame"""
        if self._amplitude_per_frame is None:
            times, values = self.analyzer.get_amplitude_envelope()
            
            # Interpolate to get value at each frame
            frame_times = np.array([self.frame_to_time(f) for f in range(self.total_frames)])
            
            # Use numpy interpolation
            self._amplitude_per_frame = np.interp(frame_times, times, values)
        
        return self._amplitude_per_frame
    
    def get_amplitude_at_frame(self, frame: int) -> float:
        """
        Get normalized amplitude (0-1) at specific frame.
        
        Args:
            frame: Frame number
            
        Returns:
            Amplitude value between 0 and 1
        """
        amplitudes = self._compute_amplitude_per_frame()
        frame = np.clip(frame, 0, len(amplitudes) - 1)
        return float(amplitudes[frame])
    
    def get_amplitude_curve(self) -> np.ndarray:
        """Get amplitude values for all frames as array"""
        return self._compute_amplitude_per_frame()
    
    # =========================================================================
    # BEATS
    # =========================================================================
    
    def _compute_beat_frames(self, tolerance: float = 0.05) -> set:
        """Pre-compute which frames fall on beats"""
        if self._beat_frames is None:
            beat_times = self.analyzer.get_beat_times()
            self._beat_frames = set()
            
            for beat_time in beat_times:
                # Find frame closest to beat
                frame = self.time_to_frame(beat_time)
                
                # Check if frame time is within tolerance of beat
                frame_time = self.frame_to_time(frame)
                if abs(frame_time - beat_time) <= tolerance:
                    self._beat_frames.add(frame)
                    
                # Also check adjacent frames (in case beat falls between frames)
                if frame > 0:
                    prev_time = self.frame_to_time(frame - 1)
                    if abs(prev_time - beat_time) <= tolerance:
                        self._beat_frames.add(frame - 1)
                        
                if frame < self.total_frames - 1:
                    next_time = self.frame_to_time(frame + 1)
                    if abs(next_time - beat_time) <= tolerance:
                        self._beat_frames.add(frame + 1)
        
        return self._beat_frames
    
    def is_beat_frame(self, frame: int, tolerance: float = 0.05) -> bool:
        """
        Check if frame falls on a beat.
        
        Args:
            frame: Frame number
            tolerance: Time tolerance in seconds
            
        Returns:
            True if frame is on a beat
        """
        beat_frames = self._compute_beat_frames(tolerance)
        return frame in beat_frames
    
    def get_beat_frames(self, tolerance: float = 0.05) -> List[int]:
        """Get list of all frame numbers that fall on beats"""
        beat_frames = self._compute_beat_frames(tolerance)
        return sorted(list(beat_frames))
    
    # =========================================================================
    # ONSETS (Transients)
    # =========================================================================
    
    def _compute_onset_frames(self, tolerance: float = 0.03) -> set:
        """Pre-compute which frames fall on onsets"""
        if self._onset_frames is None:
            onset_times = self.analyzer.get_onsets()
            self._onset_frames = set()
            
            for onset_time in onset_times:
                frame = self.time_to_frame(onset_time)
                frame_time = self.frame_to_time(frame)
                if abs(frame_time - onset_time) <= tolerance:
                    self._onset_frames.add(frame)
        
        return self._onset_frames
    
    def is_onset_frame(self, frame: int, tolerance: float = 0.03) -> bool:
        """Check if frame falls on an onset/transient"""
        onset_frames = self._compute_onset_frames(tolerance)
        return frame in onset_frames
    
    def get_onset_frames(self, tolerance: float = 0.03) -> List[int]:
        """Get list of all frame numbers that fall on onsets"""
        onset_frames = self._compute_onset_frames(tolerance)
        return sorted(list(onset_frames))
    
    # =========================================================================
    # SCHEDULE GENERATION
    # =========================================================================
    
    def generate_modifiers(self, config: MusicSyncConfig) -> List[Dict[str, float]]:
        """
        Generate per-frame parameter modifiers based on config.
        
        Args:
            config: MusicSyncConfig specifying how audio affects parameters
            
        Returns:
            List of dicts, one per frame, with parameter modifiers
            
        Example output:
            [
                {'frame': 0, 'strength_mod': 0.0, 'zoom_mod': 0.0, ...},
                {'frame': 1, 'strength_mod': 0.05, 'zoom_mod': 0.1, ...},
                ...
            ]
        """
        modifiers = []
        amplitudes = self.get_amplitude_curve()
        
        # Optional smoothing
        if config.amplitude_smoothing > 0:
            kernel_size = config.amplitude_smoothing
            kernel = np.ones(kernel_size) / kernel_size
            amplitudes = np.convolve(amplitudes, kernel, mode='same')
        
        beat_frames = self._compute_beat_frames(config.beat_tolerance)
        onset_frames = self._compute_onset_frames(config.onset_tolerance)
        
        for frame in range(self.total_frames):
            mod = {'frame': frame}
            amp = amplitudes[frame]
            is_beat = frame in beat_frames
            is_onset = frame in onset_frames
            
            # --- Amplitude-based modifiers ---
            
            if config.amplitude_to_strength:
                min_val, max_val = config.amplitude_to_strength
                mod['strength'] = min_val + amp * (max_val - min_val)
            
            if config.amplitude_to_noise:
                min_val, max_val = config.amplitude_to_noise
                mod['noise'] = min_val + amp * (max_val - min_val)
            
            if config.amplitude_to_guidance:
                min_val, max_val = config.amplitude_to_guidance
                mod['guidance_scale'] = min_val + amp * (max_val - min_val)
            
            # --- Beat-based modifiers (additive) ---
            
            mod['is_beat'] = is_beat
            mod['is_onset'] = is_onset
            
            if is_beat:
                if config.beat_strength_boost:
                    mod['strength_boost'] = config.beat_strength_boost
                if config.beat_zoom_boost:
                    mod['zoom_boost'] = config.beat_zoom_boost
                if config.beat_noise_boost:
                    mod['noise_boost'] = config.beat_noise_boost
            
            # --- Onset-based modifiers ---
            
            if is_onset:
                if config.onset_strength_boost:
                    mod['strength_boost'] = mod.get('strength_boost', 0) + config.onset_strength_boost
                if config.onset_zoom_boost:
                    mod['zoom_boost'] = mod.get('zoom_boost', 0) + config.onset_zoom_boost
            
            # Store raw amplitude for custom use
            mod['amplitude'] = float(amp)
            
            modifiers.append(mod)
        
        return modifiers
    
    def summary(self) -> str:
        """Get summary of music schedule"""
        bpm = self.analyzer.get_bpm()
        beat_frames = self.get_beat_frames()
        onset_frames = self.get_onset_frames()
        
        return f"""
Music Schedule Summary
======================
FPS: {self.fps}
Total frames: {self.total_frames}
Animation duration: {self.total_frames / self.fps:.2f}s
Audio duration: {self.analyzer.duration:.2f}s
BPM: {bpm:.1f}
Frames on beats: {len(beat_frames)}
Frames on onsets: {len(onset_frames)}
"""


# =============================================================================
# HELPER: Apply modifiers to existing schedule
# =============================================================================

def apply_music_to_schedule(
    base_schedule: List[Dict], 
    music_modifiers: List[Dict],
    blend_mode: str = 'replace'
) -> List[Dict]:
    """
    Apply music modifiers to an existing animation schedule.
    
    Args:
        base_schedule: Output from AnimationInterpolator.generate_animation_schedule()
        music_modifiers: Output from MusicSchedule.generate_modifiers()
        blend_mode: 
            'replace' - music values replace base values
            'add' - music values add to base values
            'multiply' - music values multiply base values
    
    Returns:
        Modified schedule with music-reactive parameters
    """
    if len(base_schedule) != len(music_modifiers):
        raise ValueError(f"Schedule length mismatch: {len(base_schedule)} vs {len(music_modifiers)}")
    
    result = []
    
    for base, mod in zip(base_schedule, music_modifiers):
        frame_params = base.copy()
        
        # Apply direct replacements (amplitude-mapped values)
        for key in ['strength', 'noise', 'guidance_scale']:
            if key in mod:
                if blend_mode == 'replace':
                    frame_params[key] = mod[key]
                elif blend_mode == 'add':
                    frame_params[key] = frame_params.get(key, 0) + mod[key]
                elif blend_mode == 'multiply':
                    frame_params[key] = frame_params.get(key, 1) * mod[key]
        
        # Apply boosts (always additive)
        if 'strength_boost' in mod:
            frame_params['strength'] = frame_params.get('strength', 0.5) + mod['strength_boost']
        
        if 'zoom_boost' in mod:
            frame_params['zoom'] = frame_params.get('zoom', 1.0) + mod['zoom_boost']
        
        if 'noise_boost' in mod:
            frame_params['noise'] = frame_params.get('noise', 0) + mod['noise_boost']
        
        # Copy metadata
        frame_params['is_beat'] = mod.get('is_beat', False)
        frame_params['is_onset'] = mod.get('is_onset', False)
        frame_params['amplitude'] = mod.get('amplitude', 0)
        
        result.append(frame_params)
    
    return result


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

def example_usage():
    """Example of how to use MusicSchedule"""
    print("=" * 60)
    print("MUSIC SCHEDULE - Example Usage")
    print("=" * 60)
    
    # Create test audio
    from audio_analyzer import AudioAnalyzer
    import os
    
    # Check if test audio exists from previous run
    test_file = "/tmp/test_audio.wav"
    
    if not os.path.exists(test_file):
        print("\nCreating test audio...")
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # 120 BPM = beat every 0.5 seconds
        beat_times = np.arange(0, duration, 0.5)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        
        for beat_time in beat_times:
            beat_idx = int(beat_time * sr)
            impulse_len = int(0.05 * sr)
            if beat_idx + impulse_len < len(audio):
                decay = np.exp(-np.linspace(0, 5, impulse_len))
                audio[beat_idx:beat_idx + impulse_len] += decay * 0.7
        
        import soundfile as sf
        sf.write(test_file, audio, sr)
    
    # Load audio
    analyzer = AudioAnalyzer(test_file)
    analyzer.load()
    
    # Create music schedule
    fps = 24
    total_frames = int(5.0 * fps)  # 5 seconds at 24fps = 120 frames
    
    music = MusicSchedule(analyzer, fps=fps, total_frames=total_frames)
    
    print(music.summary())
    
    # Show beat frames
    beat_frames = music.get_beat_frames()
    print(f"Beat frames: {beat_frames[:15]}...")
    
    # Create config
    config = MusicSyncConfig(
        amplitude_to_strength=(0.5, 0.8),  # strength: 0.5 when quiet, 0.8 when loud
        beat_zoom_boost=0.05,              # zoom +0.05 on beats
        beat_strength_boost=0.1,           # strength +0.1 on beats
    )
    
    # Generate modifiers
    modifiers = music.generate_modifiers(config)
    
    print("\nSample modifiers (every 12 frames):")
    print("-" * 60)
    for i in range(0, min(60, len(modifiers)), 12):
        mod = modifiers[i]
        beat_marker = "🥁" if mod.get('is_beat') else "  "
        print(f"Frame {i:3d} {beat_marker}: "
              f"amp={mod['amplitude']:.2f}, "
              f"strength={mod.get('strength', 'N/A')}, "
              f"zoom_boost={mod.get('zoom_boost', 0):.2f}")
    
    return music, modifiers


if __name__ == "__main__":
    example_usage()