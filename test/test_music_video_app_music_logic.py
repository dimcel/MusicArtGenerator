import numpy as np
from PIL import Image

from music_art_generator.music_video_app.music_logic import (
    _build_onset_twist_gain,
    _build_quiet_hold_mask,
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


def test_short_runs_are_removed():
    mask = np.array([False, True, False, True, True, False], dtype=bool)
    cleaned = _remove_short_true_runs(mask, min_len=2)
    assert cleaned.tolist() == [False, False, False, True, True, False]


def test_quiet_and_twist_masks_are_well_formed():
    energy = np.array([0.4, 0.45, 0.5, 0.1, 0.42, 0.46, 0.48], dtype=np.float32)
    onset = np.array([0.05, 0.03, 0.04, 0.7, 0.02, 0.03, 0.04], dtype=np.float32)

    quiet_mask, quiet_meta = _build_quiet_hold_mask(energy_curve=energy, onset_curve=onset, fps=24)
    assert quiet_mask.dtype == bool
    assert quiet_mask.shape == energy.shape
    assert 0.0 <= quiet_meta["quiet_frame_ratio"] <= 1.0

    twist_gain, twist_meta = _build_onset_twist_gain(onset_curve=onset, fps=24)
    assert twist_gain.shape == onset.shape
    assert 0.0 <= float(twist_gain.min()) <= float(twist_gain.max()) <= 1.0
    assert 0.0 <= twist_meta["active_frame_ratio"] <= 1.0
