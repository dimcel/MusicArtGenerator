import json
import time

from music_art_generator.music_video_app.resume import (
    _changed_run_arg_keys,
    _find_last_frame,
    _load_resume_state,
    _save_resume_state,
    _state_path,
)


def test_resume_state_save_and_load_roundtrip(tmp_path):
    run_args = {"audio_path": "song.wav", "fps": 24, "mode": "full", "coherence": "blend", "prompt_mode": "blend", "use_controlnet": False}
    args_slug = "testslug"
    state_file = _state_path(tmp_path, args_slug)

    _save_resume_state(
        state_file=state_file,
        run_args=run_args,
        args_slug=args_slug,
        last_completed_frame=12,
        seed_state=123.5,
        total_frames=240,
        fps=24,
        prompt_state={"active_prompt_idx": 1, "beat_events_seen_for_prompt": 4},
        cli_args=["--audio", "song.wav"],
    )

    loaded = _load_resume_state(tmp_path, args_slug)
    assert loaded is not None
    assert loaded["last_completed_frame"] == 12
    assert loaded["seed_state"] == 123.5
    assert loaded["prompt_state"]["active_prompt_idx"] == 1
    assert loaded["_state_file"].endswith(state_file.name)

def test_resume_multi_file_selection_prefers_most_advanced(tmp_path):
    s1 = tmp_path / "resume_state_a.json"
    s2 = tmp_path / "resume_state_b.json"

    s1.write_text(json.dumps({"last_completed_frame": 5}), encoding="utf-8")
    time.sleep(0.01)
    s2.write_text(json.dumps({"last_completed_frame": 9}), encoding="utf-8")

    loaded = _load_resume_state(tmp_path, "missing-slug")
    assert loaded is not None
    assert loaded["last_completed_frame"] == 9
    assert loaded["_state_file"].endswith("resume_state_b.json")
    assert sorted(loaded["_state_file_candidates"]) == ["resume_state_a.json", "resume_state_b.json"]


def test_changed_keys_and_last_frame_scan(tmp_path):
    (tmp_path / "frame_00000.png").write_bytes(b"x")
    (tmp_path / "frame_00010.png").write_bytes(b"x")
    (tmp_path / "frame_invalid.png").write_bytes(b"x")

    assert _find_last_frame(tmp_path) == 10

    previous = {"a": 1, "b": 2, "c": 3}
    current = {"a": 1, "b": 9, "d": 4}
    assert _changed_run_arg_keys(previous, current) == ["b", "c", "d"]
