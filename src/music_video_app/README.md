# Music Video App CLI Reference

`music_video_app` is the package-level runner for the music-reactive video pipeline.

It handles:

- audio feature extraction (beat, onset, energy, brightness, pitch)
- frame-by-frame control mapping (zoom/pan/angle/strength/cfg/noise)
- diffusion loop (txt2img first frame, img2img for subsequent frames)
- optional ControlNet canny conditioning
- optional color coherence and music color FX
- resume/restart from saved frame state
- export to MP4 and optional audio mux

## Entrypoints

Preferred:

```bash
python -m src.music_video_app --audio "your_audio.wav" ...
```

Legacy-compatible wrapper:

```bash
python test/test_real_music_feature_video.py --audio "your_audio.wav" ...
```

## Installation

This repository is not currently published as an installable Python package.
For the main setup and first-run instructions, see the
[project README](../../README.md).

### System prerequisite

- `ffmpeg` in `PATH` for video export/audio mux.

### Runtime baseline (`requirements-temp.txt`)

#### with `uv`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-temp.txt
```

#### with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-temp.txt
```

### Optional extras (`requirements-optional.txt`)

Install only if you need these optional paths:

- FILM interpolation dependencies (`tensorflow`, `tensorflow-hub`)
- optional acceleration extras (`xformers`)

```bash
uv pip install -r requirements-optional.txt
# or
pip install -r requirements-optional.txt
```

For ready-to-run configurations, see the [examples guide](../../examples/README.md).

## Runtime And Output

### Output directory and artifacts

Without resume mode, the output directory follows this pattern:

- `output_<mode>_<timestamp>_<frames>f/`

Inside it, you will get:

- frame images: `frame_00000.png`, `frame_00001.png`, ...
- resume state: `resume_state_<args_slug>.json`
- run manifest: `output_<mode>_<timestamp>_<frames>f.json`
- silent video: `output_<mode>_<timestamp>_<frames>f.mp4`
- audio-muxed video when `--with-audio` is used: `..._with_audio.mp4`

If a name already exists, a numeric suffix is added instead of overwriting the
existing output.

### Resume behavior

When `--resume-dir` is provided:

- state is loaded from matching `resume_state_<args_slug>.json`
- if exact slug is missing, runner can pick the most advanced available state
- `--max-seconds` is treated as **additional** duration to append
- argument drift is allowed; changed args are reported in logs
- a new manifest and video export are created inside the same output directory

## Known Current Behavior

- `--prompt-mode` is kept for CLI/state compatibility, but prompt construction strategy switching is not fully active yet.
- Subject-transition controller state is persisted in resume files, but transition stepping into prompt behavior is not fully wired yet.

## Complete CLI Reference

All flags from `build_parser()` in `src/music_video_app/cli.py` are listed below.

### 1) Input / Output / Run Horizon

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--audio` | `str` | required | - | Input audio file path. | `--audio "track.wav"` |
| `--fps` | `int` | `24` | - | Frames per second used for feature alignment and render pacing. | `--fps 30` |
| `--device` | `str` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `cuda:<index>` | Compute device for diffusion pipelines. `auto` prefers CUDA, then MPS, then CPU. | `--device cuda:0` |
| `--max-seconds` | `float` | `12.0` | - | Without resume: clip from start. With resume: append duration. `<=0` means full audio. | `--max-seconds 20` |
| `--with-audio` | flag | `False` | - | Mux original audio into final MP4 using ffmpeg. | `--with-audio` |
| `--resume-dir` | `str` | `None` | - | Resume from an existing output directory/state. | `--resume-dir output_cadence_20260823_153000_480f` |

### 2) Generation Mode / Cadence / Coherence

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--mode` | `str` | `full` | `full`, `cadence` | `full`: diffuse every frame. `cadence`: diffuse on cadence/beat/transition triggers. | `--mode cadence` |
| `--cadence` | `int` | `3` | - | In cadence mode, diffuse every Nth frame (plus beat triggers). | `--cadence 2` |
| `--coherence` | `str` | `blend` | `none`, `blend`, `optical_flow`, `rife`, `film` | Coherence strategy when diffusion is skipped on cadence frames. | `--coherence none` |

### 3) Prompt / Concept / Identity

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--prompt` | `str` | `""` | - | Main prompt. Supports single string, `||` list, or JSON list string. | `--prompt "scene A || scene B"` |
| `--prompt-change-every-beats` | `int` | `1` | - | For prompt lists, switch prompt every N beat events. | `--prompt-change-every-beats 2` |
| `--prompt-mode` | `str` | `blend` | `blend`, `hard` | Compatibility flag for prompt behavior mode selection. | `--prompt-mode hard` |
| `--init-image` | `str` | `None` | - | Seed frame 0 from this image (resized to pipeline dimensions). | `--init-image "starting_image.png"` |
| `--concept-mode` | `str` | `identity` | `identity`, `transform` | How strongly to preserve init-image identity over time. | `--concept-mode transform` |
| `--identity-prompt` | `str` | `""` | - | Extra identity anchor text appended when identity lock applies. | `--identity-prompt "same face and hairstyle"` |

### 4) Music-Reactive Motion / Dynamics

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--zoom-base` | `float` | `1.002` | - | Base per-frame zoom multiplier before music boosts. | `--zoom-base 1.01` |
| `--zoom-beat-boost` | `float` | `0.02` | - | Extra zoom influence from beat pulse. | `--zoom-beat-boost 0.00` |
| `--music-color-fx` | flag | `False` | - | Apply beat/energy-driven color and contrast effect. | `--music-color-fx` |
| `--onset-jitter` | flag | `False` | - | Add onset-triggered deterministic jitter to camera deltas. | `--onset-jitter` |
| `--quiet-hold` | flag | `False` | - | Detect quieter sections and calm motion/jitter/prompt changes. | `--quiet-hold` |
| `--steady-shift` | flag | `False` | - | Enable strong directional pan in stable musical sections. | `--steady-shift` |
| `--steady-activation-mode` | `str` | `auto` | `auto`, `manual` | Activation logic for steady shift. | `--steady-activation-mode auto` |
| `--steady-activation-ratio` | `float` | `1.0` | - | Auto mode: fraction of eligible stable runs to keep. | `--steady-activation-ratio 0.8` |
| `--steady-shift-probability` | `float` | `1.0` | - | Auto mode: per-run probability for applying shift. | `--steady-shift-probability 0.9` |
| `--steady-min-seconds` | `float` | `1.0` | - | Manual mode: minimum consecutive stable duration before activation. | `--steady-min-seconds 1.5` |
| `--steady-threshold` | `float` | `0.72` | - | Manual mode: stability threshold in `[0,1]`. | `--steady-threshold 0.8` |
| `--steady-shift-pixels` | `float` | `6.0` | - | Base pan amount during steady-shift activity. | `--steady-shift-pixels 20` |
| `--steady-twist` | flag | `False` | - | Enable onset-reactive rightward twist pulse. | `--steady-twist` |
| `--steady-twist-max-deg` | `float` | `4.0` | - | Max extra angle from steady twist pulse. | `--steady-twist-max-deg 4.0` |
| `--disable-camera-motion` | flag | `False` | - | Disable zoom/pan/rotate transform stage entirely. | `--disable-camera-motion` |
| `--disable-music-change` | flag | `False` | - | Freeze music-driven control changes to fixed controls. | `--disable-music-change` |

### 5) ControlNet / Canny Controls

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--controlnet` | flag | `False` | - | Enable ControlNet img2img with canny conditioning. | `--controlnet` |
| `--controlnet-model` | `str` | `lllyasviel/sd-controlnet-canny` | - | Model ID for ControlNet canny branch. | `--controlnet-model "lllyasviel/sd-controlnet-canny"` |
| `--canny-low` | `int` | `100` | - | Lower Canny threshold for control image edges. | `--canny-low 80` |
| `--canny-high` | `int` | `200` | - | Upper Canny threshold for control image edges. | `--canny-high 220` |
| `--control-scale-base` | `float` | `0.8` | - | Base ControlNet conditioning scale. | `--control-scale-base 0.9` |
| `--control-scale-beat-boost` | `float` | `0.3` | - | Control scale boost from beat pulse. | `--control-scale-beat-boost 0.25` |
| `--control-scale-onset-boost` | `float` | `0.2` | - | Control scale boost from onset activity. | `--control-scale-onset-boost 0.15` |

### 6) Color / Re-Anchor / Compatibility Transition Knobs

| Flag | Type | Default | Choices | Practical meaning | Example |
|---|---|---:|---|---|---|
| `--color-coherence` | `str` | `none` | `none`, `match_frame0_lab` | Post-process color matching against frame 0. | `--color-coherence match_frame0_lab` |
| `--color-coherence-strength` | `float` | `0.6` | - | Blend strength for color coherence correction. | `--color-coherence-strength 0.7` |
| `--re-anchor` | flag | `False` | - | Periodically blend back toward transformed frame 0. | `--re-anchor` |
| `--re-anchor-strength` | `str` | `mid` | `small`, `mid`, `big`, numeric alpha | Re-anchor profile preset or direct alpha value. | `--re-anchor-strength big` |
| `--re-anchor-every-frames` | `int` | `12` | - | Frequency for applying re-anchor logic. | `--re-anchor-every-frames 8` |
| `--subject-hold-frames` | `int` | `32` | - | Compatibility transition hold parameter (persisted in state). | `--subject-hold-frames 48` |
| `--subject-transition-frames` | `int` | `16` | - | Compatibility transition duration parameter (persisted in state). | `--subject-transition-frames 24` |
| `--subject-smoothing-window` | `int` | `41` | - | Compatibility smoothing-window parameter for subject drive. | `--subject-smoothing-window 31` |

## Example Presets

The [examples guide](../../examples/README.md) contains basic cadence, onset
jitter, quiet hold, twist, prompt-list, ControlNet, and resume commands.

## Troubleshooting

- `No module named ...`: install baseline requirements first (`requirements-temp.txt`).
- `ffmpeg mux failed`: install `ffmpeg` and verify `ffmpeg -version` works.
- `Audio file not found` / `Init image not found`: verify relative paths from your current working directory.
- Slow generation on CPU: this is expected; GPU-enabled torch is recommended for practical runtime.

## Sanity Checks

```bash
python -m src.music_video_app --help
python test/test_real_music_feature_video.py --help
```
