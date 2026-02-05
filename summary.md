# Music Art Generator - Project Summary

## Overview

Music Art Generator creates animated videos by generating AI images synchronized to music. Currently undergoing a complete refactoring to create a modular, testable codebase following best practices. The new architecture uses Deforum-style frame-by-frame generation with optional frame interpolation for faster output.

---

## Project Architecture

```
MusicArtGenerator/
├── _src/                    # ✨ NEW: Refactored core modules (clean, testable)
│   ├── image_generator.py   # Stable Diffusion wrapper
│   ├── frame_interpolator.py # Frame interpolation (blend, optical_flow, film)
│   ├── video_animator.py    # Basic Deforum loop
│   └── _video_animator.py   # Advanced: keyframe + interpolation
│
├── test_*.py                # Test files for each module
├── frames_to_video.py       # Utility: convert frames to video
│
├── src/                     # OLD: Original modules (to be refactored)
│   ├── audio_analyzer.py
│   ├── fixed_beat_schedule.py
│   ├── music_schedule.py
│   └── sd_animator.py       # 739 lines - needs refactoring
│
└── run_*.py                 # Various runners (old + new)
```

---

## Core Modules (_src/)

### 1. `image_generator.py` - Stable Diffusion Wrapper

**Purpose:** Clean interface to SD image generation

**Key Functions:**
- `generate_from_text(prompt, seed)` → txt2img for first frame
- `generate_from_image(init_image, prompt, strength)` → img2img for subsequent frames

**Features:**
- Lazy loading of pipelines (only loads when needed)
- CUDA optimization
- VAE support (stabilityai/sd-vae-ft-mse)
- No external dependencies on other project modules

**Example:**
```python
from _src.image_generator import ImageGenerator, ImageGenerationConfig

config = ImageGenerationConfig(width=512, height=512)
gen = ImageGenerator(config)

# First frame
frame_0 = gen.generate_from_text("old man on bench", seed=42)

# Next frame (img2img with noise)
frame_1 = gen.generate_from_image(frame_0, "old man on bench", strength=0.6)
```

---

### 2. `frame_interpolator.py` - Frame Interpolation

**Purpose:** Generate in-between frames to reduce diffusion cost

**Methods Available:**

| Method | Status | Quality | Speed | Notes |
|--------|--------|---------|-------|-------|
| `blend` | ✅ Working | Low | Very fast | Simple alpha blending, ghosting effect |
| `optical_flow` | ✅ Working | Good | Fast | OpenCV flow warping, motion-aware |
| `film` | ⚠️ Needs fix | Best | Slow | Google neural model, TensorFlow required |
| `rife` | ❌ Stub only | Very good | Real-time | Needs manual implementation |

**How It Works:**
- Takes 2 keyframes (frame_a, frame_b)
- Generates N frames in between
- Returns list of interpolated frames

**Example:**
```python
from _src.frame_interpolator import FrameInterpolator

interpolator = FrameInterpolator(method="optical_flow")
in_between = interpolator.interpolate(frame_0, frame_4, num_frames=3)
# Returns: [frame_1, frame_2, frame_3]
```

**Speedup:**
- Interval=2 → 2x faster (generate every 2nd frame)
- Interval=4 → 4x faster (generate every 4th frame)
- Interval=8 → 8x faster (generate every 8th frame)

---

### 3. `video_animator.py` vs `_video_animator.py`

**`video_animator.py` (Original - Simple)**
- No interpolation support
- Generates all frames with diffusion
- Slower but maximum quality
- Use for: Final quality output

**`_video_animator.py` (New - Advanced)**
- ✅ Interpolation support (optional)
- Two-pass system:
  - PASS 1: Generate keyframes only
  - PASS 2: Interpolate in-between frames
- Configurable: method, interval, strength
- Use for: Fast iteration, testing

**Configuration Options:**
```python
from _src._video_animator import VideoAnimator, VideoAnimationConfig

config = VideoAnimationConfig(
    prompt="old man sitting on a bench",
    total_frames=200,
    fps=24,
    strength=0.6,                      # Noise amount per keyframe
    use_interpolation=True,            # Enable interpolation
    keyframe_interval=4,               # Generate every 4th frame
    interpolation_method="optical_flow", # blend, optical_flow, film
    output_dir="my_animation"
)

animator = VideoAnimator(config)
video_path = animator.generate_and_save_video()
```

---

## Testing Infrastructure

### Test Files

**`test_image_generator.py`**
- Tests: txt2img, img2img, strength variations
- Verifies SD pipeline works correctly

**`test_video_animator.py`**
- Tests: Basic video generation without interpolation
- Compares different strength values

**`test_interpolation.py`** ⭐ Main test file
- Test 1: Standalone interpolator (quick test)
- Test 2: Baseline video (no interpolation)
- Test 3: Video with interpolation
- Test 4: Interval comparison (2x, 4x, 8x speedup)
- Test 5: Simple helper function

**Global Configuration:**
```python
# At top of test_interpolation.py
TOTAL_FRAMES = 200  # Change here to control all test video lengths
```

**Run Tests:**
```bash
python test_interpolation.py          # All tests
python test_interpolation.py --test 3 # Just test 3
```

---

## Current Status

### ✅ Completed Features

- Image generation module (txt2img, img2img)
- Frame interpolation with 3 methods (blend, optical_flow, film)
- Keyframe-based video generation
- Interpolation speedup (2x-8x faster)
- Comprehensive test suite
- Gap-tolerant video creation (concat demuxer)
- Global configuration for easy testing
- Utilities: frames_to_video.py

### ⚠️ Known Issues

- **FILM interpolation:** Tensor shape mismatch (falls back to optical_flow)
- **RIFE interpolation:** Not implemented (stub only)
- **Old codebase:** Still needs refactoring (src/ folder)

### 📊 Performance

**Generation Speed (200 frames at 512x512):**
- No interpolation: ~200 frames × 3s = **10 minutes**
- Interval=4: ~50 keyframes × 3s = **2.5 minutes** (4x faster)
- Interval=8: ~25 keyframes × 3s = **1.25 minutes** (8x faster)

---

## Next Steps

### 1. Music Integration 🎵

**Goal:** Add beat/audio signals to control generation parameters

**Approach:**
- Detect beats from audio file (librosa)
- Create beat schedule (already exists: `src/fixed_beat_schedule.py`)
- Inject signal into generation process:
  - Increase `strength` on beats (more change)
  - Decrease `strength` between beats (more stable)
  - Dynamic keyframe placement (more keyframes on beats)

**Where to integrate:**
```python
# In _video_animator.py, modify PASS 1:
for frame_num in keyframe_nums:
    if beat_schedule.is_beat(frame_num):
        strength = 0.8  # High change on beat
    else:
        strength = 0.4  # Stable between beats
    
    current_image = generator.generate_from_image(
        init_image=current_image,
        strength=strength,  # Dynamic strength
        ...
    )
```

**Implementation steps:**
- Refactor `src/audio_analyzer.py` into `_src/`
- Add beat detection to `VideoAnimationConfig`
- Modify keyframe generation to use dynamic strength
- Test with music synchronization

---

### 2. Frame Transformations 🎬

**Goal:** Add Deforum-style camera movements (zoom, pan, rotate)

**Deforum Effects:**
- **Zoom:** Scale image in/out
- **Pan:** Translate left/right/up/down
- **Rotate:** 2D rotation
- **3D rotation:** Perspective transforms
- **Combined:** Multiple effects per frame

**Approach:**
- Create new module: `_src/frame_transformer.py`
- Apply transformations BEFORE diffusion:
  1. Take previous frame
  2. Apply zoom/pan/rotate transform
  3. Feed transformed image to img2img
  4. Diffusion fills in missing areas

**Example workflow:**
```
Frame 0: Generate from text
Frame 1: 
  → Take frame 0
  → Zoom in 5%
  → Pan right 10px
  → Rotate 2°
  → Feed to img2img (strength=0.6)
  → Diffusion fills edges
Frame 2: Repeat with frame 1
```

**Where this fits:**
```
Current pipeline:
  previous_image → img2img → new_image

New pipeline:
  previous_image → transform (zoom/pan/rotate) → img2img → new_image
                   ^^^^^^^^^^^^^^^^^^^^^^^^
                   New: frame_transformer.py
```

**Implementation steps:**
- Create `_src/frame_transformer.py`
- Add transformation config (zoom_per_frame, pan_x, pan_y, angle)
- Integrate into `_video_animator.py` before img2img
- Add to `VideoAnimationConfig`

---

## Quick Start

### Basic Usage

```python
from _src._video_animator import simple_animation

# Fast mode with interpolation
video = simple_animation(
    prompt="old man on bench, peaceful park",
    frames=200,
    strength=0.6,
    use_interpolation=True,
    keyframe_interval=4,
    interpolation_method="optical_flow"
)
```

### Advanced Usage

```python
from _src._video_animator import VideoAnimator, VideoAnimationConfig

config = VideoAnimationConfig(
    prompt="cyberpunk city street, neon lights, rainy night",
    total_frames=240,  # 10 seconds at 24fps
    fps=24,
    strength=0.7,
    use_interpolation=True,
    keyframe_interval=4,
    interpolation_method="optical_flow",
    output_dir="cyberpunk_animation",
    width=768,
    height=512,
    seed=42
)

animator = VideoAnimator(config)
video_path = animator.generate_and_save_video()
print(f"Video saved: {video_path}")
```

### Testing

```bash
# Test interpolation system
python test_interpolation.py --test 3

# Convert existing frames to video
python frames_to_video.py --frames_dir output_frames --fps 24
```

---

## Key Design Principles

1. **Single Responsibility:** Each module does one thing well
2. **Lazy Loading:** Only load models when needed
3. **Testability:** Every module has tests
4. **Configurability:** Dataclass configs for all parameters
5. **Fallback Support:** Graceful degradation if dependencies missing
6. **Simple API:** Easy-to-use helper functions alongside advanced options

---

## Development Workflow

1. **Current Phase:** Refactoring core modules into `_src/`
2. **Testing:** Each module tested independently
3. **Integration:** Modules work together through clean interfaces
4. **Next Phase:** Add music sync and frame transformations
5. **Future:** Complete refactoring of `src/` folder

---

*Last updated: Session refactoring (feat/new_model branch)*
