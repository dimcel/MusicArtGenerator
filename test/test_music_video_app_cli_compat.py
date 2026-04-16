import pytest

from src.music_video_app.cli import build_parser


EXPECTED_FLAGS = {
    "--audio",
    "--fps",
    "--mode",
    "--cadence",
    "--prompt",
    "--prompt-change-every-beats",
    "--music-color-fx",
    "--onset-jitter",
    "--quiet-hold",
    "--color-coherence",
    "--color-coherence-strength",
    "--prompt-mode",
    "--coherence",
    "--subject-hold-frames",
    "--subject-transition-frames",
    "--subject-smoothing-window",
    "--max-seconds",
    "--with-audio",
    "--controlnet",
    "--controlnet-model",
    "--control-scale-base",
    "--control-scale-beat-boost",
    "--control-scale-onset-boost",
    "--zoom-base",
    "--zoom-beat-boost",
    "--steady-shift",
    "--steady-activation-mode",
    "--steady-activation-ratio",
    "--steady-shift-probability",
    "--steady-min-seconds",
    "--steady-threshold",
    "--steady-shift-pixels",
    "--steady-twist",
    "--steady-twist-max-deg",
    "--canny-low",
    "--canny-high",
    "--init-image",
    "--concept-mode",
    "--identity-prompt",
    "--re-anchor",
    "--re-anchor-strength",
    "--re-anchor-every-frames",
    "--disable-camera-motion",
    "--disable-music-change",
    "--resume-dir",
}


def test_cli_flags_match_legacy_set():
    parser = build_parser()
    flags = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert flags == EXPECTED_FLAGS


def test_cli_defaults_and_types_match_legacy_behavior():
    parser = build_parser()
    args = parser.parse_args(["--audio", "song.wav"])

    assert args.audio == "song.wav"
    assert args.fps == 24 and isinstance(args.fps, int)
    assert args.mode == "full"
    assert args.cadence == 3
    assert args.prompt == ""
    assert args.prompt_change_every_beats == 1
    assert args.music_color_fx is False
    assert args.onset_jitter is False
    assert args.quiet_hold is False
    assert args.color_coherence == "none"
    assert args.color_coherence_strength == 0.60 and isinstance(args.color_coherence_strength, float)
    assert args.prompt_mode == "blend"
    assert args.coherence == "blend"
    assert args.subject_hold_frames == 32
    assert args.subject_transition_frames == 16
    assert args.subject_smoothing_window == 41
    assert args.max_seconds == 12.0
    assert args.with_audio is False
    assert args.controlnet is False
    assert args.controlnet_model == "lllyasviel/sd-controlnet-canny"
    assert args.control_scale_base == 0.80
    assert args.control_scale_beat_boost == 0.30
    assert args.control_scale_onset_boost == 0.20
    assert args.zoom_base == 1.002
    assert args.zoom_beat_boost == 0.020
    assert args.steady_shift is False
    assert args.steady_activation_mode == "auto"
    assert args.steady_activation_ratio == 1.0
    assert args.steady_shift_probability == 1.0
    assert args.steady_min_seconds == 1.0
    assert args.steady_threshold == 0.72
    assert args.steady_shift_pixels == 6.0
    assert args.steady_twist is False
    assert args.steady_twist_max_deg == 4.0
    assert args.canny_low == 100
    assert args.canny_high == 200
    assert args.init_image is None
    assert args.concept_mode == "identity"
    assert args.identity_prompt == ""
    assert args.re_anchor is False
    assert args.re_anchor_strength == "mid"
    assert args.re_anchor_every_frames == 12
    assert args.disable_camera_motion is False
    assert args.disable_music_change is False
    assert args.resume_dir is None


def test_cli_requires_audio_argument():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
