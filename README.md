# MusicArtGenerator

MusicArtGenerator creates music-reactive videos by turning an audio track into frame-by-frame visual controls (camera motion, prompt shifts, diffusion strength, optional ControlNet), then exporting an MP4 and optionally muxing audio.

## Entrypoints

- Preferred package entrypoint: `python -m src.music_video_app ...`
- Legacy-compatible entrypoint: `python test/test_real_music_feature_video.py ...`

For the complete CLI reference (all arguments), see [`src/music_video_app/README.md`](src/music_video_app/README.md).

## Prerequisites

- Python 3.10+
- `ffmpeg` available in `PATH` (required for video export/audio mux)

Quick `ffmpeg` install examples:

- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows (winget): `winget install Gyan.FFmpeg`

## Install Dependencies

This repo currently uses **temporary requirements files**:

- `requirements-temp.txt` -> runtime baseline
- `requirements-optional.txt` -> optional extras (FILM/RIFE-related paths and perf extras)

### Option A: `uv`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-temp.txt
# optional extras
uv pip install -r requirements-optional.txt
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-temp.txt
# optional extras
pip install -r requirements-optional.txt
```

## Minimum Run

Minimum required + core controls:

- `--audio`: input audio file
- `--max-seconds`: generation duration cap
- `--mode`: `full` or `cadence`
- `--cadence`: cadence interval when in cadence mode
- `--with-audio`: mux original audio into final MP4

Example:

```bash
python -m src.music_video_app \
  --audio "fake_debug_profile.wav" \
  --max-seconds 12 \
  --mode cadence \
  --cadence 2 \
  --with-audio
```

## Most-Used Optional Args

### Prompt / Init Image

- `--prompt`
- `--init-image`
- `--concept-mode` (`identity` or `transform`)
- `--identity-prompt`

### Motion / Reactivity

- `--zoom-base`, `--zoom-beat-boost`
- `--steady-shift`, `--steady-shift-pixels`
- `--steady-activation-mode`, `--steady-activation-ratio`, `--steady-shift-probability`
- `--steady-twist`, `--steady-twist-max-deg`
- `--quiet-hold`, `--onset-jitter`

### ControlNet

- `--controlnet`
- `--controlnet-model`
- `--canny-low`, `--canny-high`
- `--control-scale-base`, `--control-scale-beat-boost`, `--control-scale-onset-boost`

### Runtime

- `--device` (`auto`, `cpu`, `mps`, `cuda`, `cuda:<index>`)

### Resume

- `--resume-dir`

## Preset Example: Twist

```bash
python -m src.music_video_app \
  --audio "fake_debug_profile.wav" \
  --init-image "../afro.png" \
  --concept-mode transform \
  --prompt "Herbie Hancock sci-fi landscape on space, Herbie Hancock keyboardist with afro hair inside a transparent spherical space cockpit, Herbie Hancock playing a futuristic synthesizer, Herbie Hancock floating above dense purple clouds, massive surreal mountain with geometric ancient city carved into it, dreamy and cosmic atmosphere, 1970s jazz album cover style, soft airbrushed textures, highly detailed, retro-futurism, warm magenta and violet color palette with purple highlights, cinematic lighting" \
  --max-seconds 12 \
  --controlnet \
  --mode cadence \
  --cadence 2 \
  --coherence none \
  --zoom-base 1.01 \
  --zoom-beat-boost 0.00 \
  --steady-activation-mode auto \
  --steady-activation-ratio 1.0 \
  --steady-shift-probability 0.9 \
  --steady-twist \
  --steady-twist-max-deg 4.0 \
  --with-audio
```

## Preset Example: Twist + Shift

```bash
python -m src.music_video_app \
  --audio "../afro.mp3" \
  --init-image "../afro.png" \
  --concept-mode transform \
  --prompt "Herbie Hancock sci-fi landscape on space, Herbie Hancock keyboardist with afro hair inside a transparent spherical space cockpit, Herbie Hancock playing a futuristic synthesizer, Herbie Hancock floating above dense purple clouds, massive surreal mountain with geometric ancient city carved into it, dreamy and cosmic atmosphere, 1970s jazz album cover style, soft airbrushed textures, highly detailed, retro-futurism, warm magenta and violet color palette with purple highlights, cinematic lighting" \
  --max-seconds 20 \
  --controlnet \
  --mode cadence \
  --cadence 2 \
  --coherence none \
  --zoom-base 1.01 \
  --zoom-beat-boost 0.00 \
  --steady-shift \
  --steady-shift-pixels 20 \
  --steady-activation-mode auto \
  --steady-activation-ratio 1.0 \
  --steady-shift-probability 0.9 \
  --steady-twist \
  --steady-twist-max-deg 4.0 \
  --with-audio
```

## Sanity Checks

```bash
python -m src.music_video_app --help
python test/test_real_music_feature_video.py --help
```
