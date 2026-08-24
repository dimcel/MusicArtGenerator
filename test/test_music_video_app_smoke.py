import json
from pathlib import Path

import numpy as np
from PIL import Image

import music_art_generator.frames_to_video as frames_to_video_module
import music_art_generator.music_video_app.runner as runner


class _FakeFeatures:
    def __init__(self, total_frames: int):
        self.bpm = 120.0
        self.beat_frames = [1] if total_frames > 1 else [0]
        self.beat_pulse = np.zeros(total_frames, dtype=np.float32)
        if total_frames > 1:
            self.beat_pulse[1] = 1.0
        self.energy = np.linspace(0.2, 0.8, total_frames, dtype=np.float32)
        self.onset = np.linspace(0.1, 0.3, total_frames, dtype=np.float32)
        self.brightness = np.linspace(0.3, 0.7, total_frames, dtype=np.float32)
        self.pitch = np.linspace(0.4, 0.6, total_frames, dtype=np.float32)


class _FakeExtractor:
    def __init__(self, audio_path: str, fps: int):
        self.audio_path = audio_path
        self.fps = fps
        self.duration_seconds = 1.0

    def load(self):
        return self

    def extract(self, total_frames: int):
        return _FakeFeatures(total_frames=total_frames)


class _FakeGenerator:
    def __init__(self, config):
        self.config = config

    def generate_from_text(self, prompt: str, seed: int):
        color = (seed % 255, 20, 40)
        return Image.new("RGB", (self.config.width, self.config.height), color=color)

    def generate_from_image(
        self,
        init_image,
        prompt,
        strength,
        guidance_scale,
        seed,
        control_image,
        controlnet_conditioning_scale,
    ):
        _ = prompt, strength, guidance_scale, seed, control_image, controlnet_conditioning_scale
        return init_image.copy()


class _FakeImageGenerationConfig:
    def __init__(self, **kwargs):
        self.width = kwargs.get("width", 512)
        self.height = kwargs.get("height", 512)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeCoherence:
    def __init__(self, method: str, blend_alpha: float):
        _ = method, blend_alpha

    def apply(self, prev_frame, transformed_frame):
        _ = prev_frame
        return transformed_frame


def test_runner_smoke_with_mocked_dependencies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _fake_frames_to_video(frames_dir, output_path, fps):
        _ = frames_dir, fps
        with open(output_path, "wb") as f:
            f.write(b"fake-video")

    monkeypatch.setattr(
        frames_to_video_module,
        "frames_to_video",
        _fake_frames_to_video,
    )

    monkeypatch.setattr(runner, "AudioFeatureExtractor", _FakeExtractor)
    monkeypatch.setattr(runner, "ImageGenerator", _FakeGenerator)
    monkeypatch.setattr(runner, "ImageGenerationConfig", _FakeImageGenerationConfig)
    monkeypatch.setattr(runner, "CadenceCoherence", _FakeCoherence)
    monkeypatch.setattr(runner, "transform_image", lambda image, **kwargs: image.copy())
    monkeypatch.setattr(runner, "add_gaussian_noise", lambda image, amount, seed: image.copy())

    runner.run(
        audio_path="song.wav",
        fps=4,
        max_seconds=1.0,
        with_audio=False,
        use_controlnet=False,
    )

    output_dirs = sorted(tmp_path.glob("output_full_*"))
    assert len(output_dirs) == 1
    out_dir = output_dirs[0]
    assert out_dir.exists()
    assert out_dir.name.endswith("_4f")

    frame_files = sorted(out_dir.glob("frame_*.png"))
    assert [p.name for p in frame_files] == [
        "frame_00000.png",
        "frame_00001.png",
        "frame_00002.png",
        "frame_00003.png",
    ]

    state_files = list(out_dir.glob("resume_state_*.json"))
    assert len(state_files) == 1
    state = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert state["last_completed_frame"] == 3

    base_name = out_dir.name
    video_path = out_dir / f"{base_name}.mp4"
    assert video_path.exists()

    manifest_path = out_dir / f"{base_name}.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output_base_name"] == base_name
    assert manifest["run_status"] == "completed"
    assert manifest["video_exists"] is True
    assert manifest["args_slug"] == state["args_slug"]


def test_resume_run_creates_new_export_and_uses_current_fps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export_calls = []

    def _fake_frames_to_video(frames_dir, output_path, fps):
        export_calls.append(
            {
                "frames_dir": Path(frames_dir),
                "output_path": Path(output_path),
                "fps": int(fps),
            }
        )
        with open(output_path, "wb") as f:
            f.write(b"fake-video")

    monkeypatch.setattr(
        frames_to_video_module,
        "frames_to_video",
        _fake_frames_to_video,
    )

    monkeypatch.setattr(runner, "AudioFeatureExtractor", _FakeExtractor)
    monkeypatch.setattr(runner, "ImageGenerator", _FakeGenerator)
    monkeypatch.setattr(runner, "ImageGenerationConfig", _FakeImageGenerationConfig)
    monkeypatch.setattr(runner, "CadenceCoherence", _FakeCoherence)
    monkeypatch.setattr(runner, "transform_image", lambda image, **kwargs: image.copy())
    monkeypatch.setattr(runner, "add_gaussian_noise", lambda image, amount, seed: image.copy())

    runner.run(
        audio_path="song.wav",
        fps=4,
        max_seconds=1.0,
        with_audio=False,
        use_controlnet=False,
    )

    output_dirs = sorted(tmp_path.glob("output_full_*"))
    assert len(output_dirs) == 1
    out_dir = output_dirs[0]
    first_video = out_dir / f"{out_dir.name}.mp4"
    assert first_video.exists()

    runner.run(
        audio_path="song.wav",
        fps=6,
        max_seconds=1.0,
        with_audio=False,
        use_controlnet=False,
        resume_dir=str(out_dir),
    )

    mp4_files = sorted(out_dir.glob("*.mp4"))
    assert len(mp4_files) == 2
    assert first_video in mp4_files
    second_video = [p for p in mp4_files if p != first_video][0]
    assert "_6f" in second_video.stem

    assert [call["fps"] for call in export_calls] == [4, 6]
    assert export_calls[0]["output_path"].resolve() == first_video.resolve()
    assert export_calls[1]["output_path"] == second_video

    manifest_files = sorted(out_dir.glob("output_full_*.json"))
    assert len(manifest_files) == 2
