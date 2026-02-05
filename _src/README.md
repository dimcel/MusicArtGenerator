# Refactored Code Structure (_src)

## Overview
Clean, step-by-step refactoring following best coding principles.

## Design Principles

### 1. **Single Responsibility Principle (SRP)**
Each module does ONE thing well.

### 2. **Keep It Simple (KISS)**
Minimal complexity, clear APIs, no unnecessary features.

### 3. **Separation of Concerns**
Clear boundaries between modules.

### 4. **Testability**
Each module can be tested independently.

---

## Module 1: Image Generator ✅

**File:** `_src/image_generator.py`

**Responsibility:** Generate images using Stable Diffusion

**Features:**
- Text-to-image (first frame)
- Image-to-image (subsequent frames)
- Simple configuration via dataclass
- Lazy loading of pipelines
- Clean memory management

**API:**
```python
from _src.image_generator import ImageGenerator, ImageGenerationConfig

# Configure
config = ImageGenerationConfig(
    width=512,
    height=512,
    num_inference_steps=30
)

# Create generator
generator = ImageGenerator(config)

# Generate first frame from text
image = generator.generate_from_text(
    prompt="old man on bench",
    seed=42
)

# Generate next frame from image
next_image = generator.generate_from_image(
    init_image=image,
    prompt="old man on bench, sunset",
    strength=0.6,
    seed=42
)
```

**Why it's better:**
- ✅ Clear separation: txt2img vs img2img
- ✅ Simple API: just two methods
- ✅ Configurable: dataclass for settings
- ✅ No dependencies on other modules
- ✅ Easy to test

---

## Testing

**Test file:** `test_image_generator.py`

Run the test:
```bash
python test_image_generator.py
```

This will:
1. Generate an image from text
2. Generate an image from the previous image
3. Test different strength values
4. Save results to `test_output/`

**For Colab:**
```python
!python test_image_generator.py

# View results
from IPython.display import Image
Image('test_output/test_txt2img.png')
```

---

## Next Steps (Coming Soon)

### Module 2: Frame Transformer
**Responsibility:** Apply camera motion (zoom, rotation, translation)
- Simple transform operations
- No complex dependencies
- Pure image manipulation

### Module 3: Animation Scheduler
**Responsibility:** Manage frame parameters and interpolation
- Keyframe interpolation
- Parameter scheduling
- No generation logic

### Module 4: Beat Detector (Optional)
**Responsibility:** Detect beats at fixed intervals or from audio
- Fixed beat timing
- Simple beat scheduling
- No animation logic

### Module 5: Animation Renderer
**Responsibility:** Orchestrate the animation generation
- Uses ImageGenerator for frames
- Uses FrameTransformer for motion
- Uses AnimationScheduler for parameters
- Simple loop: generate → transform → save

---

## Comparison: Old vs New

### Old Structure (`src/`)
```
src/
├── sd_animator.py          # Does EVERYTHING (739 lines!)
│   ├── Load models
│   ├── Generate images
│   ├── Transform images
│   ├── Manage schedules
│   ├── Handle cadence
│   ├── Create videos
│   └── ...way too much
├── audio_analyzer.py
├── music_schedule.py
├── optical_flow_cadence.py
└── ...
```

**Problems:**
- ❌ Single file does too much
- ❌ Hard to test individual features
- ❌ Tight coupling between modules
- ❌ Difficult to modify without breaking things

### New Structure (`_src/`)
```
_src/
├── image_generator.py      # ONLY generates images (250 lines)
│   ├── txt2img
│   └── img2img
│
├── frame_transformer.py    # ONLY transforms images (coming)
│   ├── zoom
│   ├── rotate
│   └── translate
│
├── animation_scheduler.py  # ONLY manages schedules (coming)
│   ├── keyframe interpolation
│   └── parameter curves
│
├── beat_detector.py        # ONLY detects beats (coming)
│   ├── fixed interval
│   └── audio analysis
│
└── animation_renderer.py   # ONLY orchestrates (coming)
    └── uses all above modules
```

**Benefits:**
- ✅ Each module has one job
- ✅ Easy to test individually
- ✅ Loose coupling
- ✅ Easy to modify or replace modules
- ✅ Clear data flow

---

## Development Process

### Phase 1: Image Generation ✅ (DONE)
- [x] Create `image_generator.py`
- [x] Create test script
- [x] Verify it works

### Phase 2: Frame Transformation (NEXT)
- [ ] Create `frame_transformer.py`
- [ ] Extract zoom/rotate/translate logic
- [ ] Create test script
- [ ] Verify transformations work

### Phase 3: Animation Scheduling
- [ ] Create `animation_scheduler.py`
- [ ] Handle keyframe interpolation
- [ ] Create test script
- [ ] Verify interpolation works

### Phase 4: Beat Detection
- [ ] Create `beat_detector.py`
- [ ] Fixed interval beats
- [ ] Create test script
- [ ] Verify beat timing works

### Phase 5: Animation Rendering
- [ ] Create `animation_renderer.py`
- [ ] Combine all modules
- [ ] Create full animation test
- [ ] Verify complete pipeline works

---

## Testing Strategy

Each module gets its own test:

```bash
# Test image generation
python test_image_generator.py

# Test frame transformation (coming)
python test_frame_transformer.py

# Test animation scheduling (coming)
python test_animation_scheduler.py

# Test beat detection (coming)
python test_beat_detector.py

# Test full rendering (coming)
python test_animation_renderer.py
```

This allows us to:
- ✅ Test each part independently
- ✅ Find bugs faster
- ✅ Verify changes don't break existing code
- ✅ Build confidence step by step

---

## Code Quality Standards

### 1. Clear Naming
```python
# Good
def generate_from_text(prompt: str) -> Image.Image:

# Bad
def gen(p):
```

### 2. Type Hints
```python
# Good
def generate_from_image(
    init_image: Image.Image,
    prompt: str,
    strength: float
) -> Image.Image:

# Bad
def generate_from_image(init_image, prompt, strength):
```

### 3. Docstrings
```python
"""
Generate image from text prompt.

Args:
    prompt: Text description
    seed: Random seed

Returns:
    Generated PIL Image
"""
```

### 4. Configuration Objects
```python
# Good - single config object
config = ImageGenerationConfig(width=512, height=512)
generator = ImageGenerator(config)

# Bad - many parameters
generator = ImageGenerator(512, 512, 7.5, 30, "cuda", ...)
```

### 5. Lazy Loading
```python
# Good - load only when needed
def _load_txt2img_pipeline(self):
    if self._txt2img_pipe is not None:
        return
    # load pipeline...

# Bad - load everything upfront
def __init__(self):
    self.txt2img = load_txt2img()  # might not need this!
    self.img2img = load_img2img()
```

---

## Summary

✅ **Module 1 Complete**: Image Generator
- Clean, simple API
- Well-tested
- Ready to use

🔄 **Next**: Frame Transformer
- Will be just as simple
- Test step by step
- Build confidence gradually

This approach ensures:
- Quality code from the start
- Easy to maintain
- Easy to extend
- Easy to understand
