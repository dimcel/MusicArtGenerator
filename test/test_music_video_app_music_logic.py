import numpy as np
from PIL import Image

from music_art_generator.music_video_app.music_logic import (
    _build_auto_steady_activation_mask,
    _build_onset_twist_gain,
    _build_quiet_hold_mask,
    _build_stability_curve,
    _build_stable_run_lengths,
    _load_and_resize_init_image,
    _parse_prompt_candidates,
    _remove_short_true_runs,
)


def test_parse_prompt_candidates_variants():
    assert _parse_prompt_candidates("") == ["cinematic music video frame"]
    assert _parse_prompt_candidates("single prompt") == ["single prompt"]
    assert _parse_prompt_candidates("a || b || c") == ["a", "b", "c"]
    assert _parse_prompt_candidates('["x", "y"]') == ["x", "y"]


def test_load_and_resize_init_image(tmp_path):
    src = tmp_path / "init.png"
    Image.new("RGB", (32, 24), color=(12, 34, 56)).save(src)

    img = _load_and_resize_init_image(str(src), width=64, height=64)
    assert img.size == (64, 64)


def test_stability_and_run_helpers():
    onset = np.array([0.1, 0.1, 0.8, 0.1, 0.1], dtype=np.float32)
    pitch = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    bright = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)

    stability = _build_stability_curve(onset, pitch, bright)
    assert stability.shape == onset.shape
    assert 0.0 <= float(stability.min()) <= float(stability.max()) <= 1.0

    run_lengths = _build_stable_run_lengths(stability, threshold=0.7)
    assert run_lengths.shape == onset.shape

    mask = np.array([False, True, False, True, True, False], dtype=bool)
    cleaned = _remove_short_true_runs(mask, min_len=2)
    assert cleaned.tolist() == [False, False, False, True, True, False]


def test_auto_steady_quiet_and_twist_masks_are_well_formed():
    stability = np.array([0.9, 0.92, 0.88, 0.2, 0.91, 0.93, 0.89], dtype=np.float32)
    energy = np.array([0.4, 0.45, 0.5, 0.1, 0.42, 0.46, 0.48], dtype=np.float32)
    onset = np.array([0.05, 0.03, 0.04, 0.7, 0.02, 0.03, 0.04], dtype=np.float32)

    active_mask, steady_meta = _build_auto_steady_activation_mask(
        stability_curve=stability,
        energy_curve=energy,
        onset_curve=onset,
        beat_frames=[0, 3, 6],
        fps=24,
        activation_ratio=1.0,
        shift_probability=1.0,
    )
    assert active_mask.dtype == bool
    assert active_mask.shape == stability.shape
    assert 0.0 <= steady_meta["actual_frame_ratio"] <= 1.0

    quiet_mask, quiet_meta = _build_quiet_hold_mask(energy_curve=energy, onset_curve=onset, fps=24)
    assert quiet_mask.dtype == bool
    assert quiet_mask.shape == energy.shape
    assert 0.0 <= quiet_meta["quiet_frame_ratio"] <= 1.0

    twist_gain, twist_meta = _build_onset_twist_gain(onset_curve=onset, fps=24)
    assert twist_gain.shape == onset.shape
    assert 0.0 <= float(twist_gain.min()) <= float(twist_gain.max()) <= 1.0
    assert 0.0 <= twist_meta["active_frame_ratio"] <= 1.0
