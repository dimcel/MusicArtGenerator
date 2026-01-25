from audio_analyzer import AudioAnalyzer

# Load and analyze
analyzer = AudioAnalyzer("song.mp3")
analyzer.load()

# Get features
bpm = analyzer.get_bpm()                    # → 120.0
beats = analyzer.get_beat_times()           # → [0.5, 1.0, 1.5, 2.0, ...]
times, amps = analyzer.get_amplitude_envelope()  # → arrays

# Query at specific times
amp = analyzer.get_amplitude_at_time(1.5)   # → 0.85
on_beat = analyzer.is_beat_near(1.0)        # → True

