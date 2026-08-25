from pathlib import Path

import music_art_generator.frames_to_video as converter


def test_frames_to_video_uses_sorted_frames_and_cleans_file_list(tmp_path, monkeypatch):
    (tmp_path / "frame_00002.png").write_bytes(b"frame two")
    (tmp_path / "frame_00000.png").write_bytes(b"frame zero")
    captured = {}

    def fake_system(command):
        captured["command"] = command
        captured["file_list"] = (tmp_path / "ffmpeg_filelist.txt").read_text()
        return 0

    monkeypatch.setattr(converter.os, "system", fake_system)
    output = tmp_path / "result.mp4"

    result = converter.frames_to_video(str(tmp_path), str(output), fps=24)

    assert result == str(output)
    assert captured["file_list"].splitlines() == [
        f"file '{(tmp_path / 'frame_00000.png').absolute()}'",
        f"file '{(tmp_path / 'frame_00002.png').absolute()}'",
    ]
    assert "-f concat" in captured["command"]
    assert not (tmp_path / "ffmpeg_filelist.txt").exists()


def test_frames_to_video_rejects_missing_directory(tmp_path):
    missing = Path(tmp_path) / "missing"
    assert converter.frames_to_video(str(missing)) is None
