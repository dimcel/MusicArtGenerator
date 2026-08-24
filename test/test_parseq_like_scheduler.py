"""
Quick demo for ParseqLikeScheduler.

Run:
    python test_parseq_like_scheduler.py
"""

from music_art_generator.parseq_like_scheduler import ParameterSpec, ParseqLikeScheduler


def fake_beats(total_frames: int, every: int = 12):
    return list(range(0, total_frames, every))


def main():
    total_frames = 48
    beats = fake_beats(total_frames, every=12)

    scheduler = ParseqLikeScheduler(
        beat_frames=beats,
        specs={
            # Keyframe curve: strength ramps over time.
            "strength": ParameterSpec(
                keyframes=[(0, 0.45), (24, 0.65), (47, 0.55)],
                easing="ease_in_out",
                beat_pulse_amount=0.15,
                beat_pulse_decay_frames=2,
                clamp=(0.3, 0.9),
            ),
            # Expression: base CFG + beat pulse boost.
            "cfg_scale": ParameterSpec(
                expression="7.0 + 1.5*sin(2*pi*t) + 2.0*beat_pulse",
                clamp=(5.0, 12.0),
            ),
            # Keyframe motion schedule.
            "zoom_delta": ParameterSpec(
                keyframes=[(0, 1.002), (24, 1.008), (47, 1.004)],
                easing="linear",
            ),
        },
    )

    print("Frame | Beat | Strength | CFG   | Zoom")
    print("-" * 40)
    for frame in range(total_frames):
        params = scheduler.frame_params(frame, total_frames)
        beat_flag = "*" if frame in beats else ""
        print(
            f"{frame:>5} | {beat_flag:^4} | "
            f"{params['strength']:.3f}    | "
            f"{params['cfg_scale']:.3f} | "
            f"{params['zoom_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
