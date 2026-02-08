# Project TODO - Music Art Generator Refactoring

## ✅ Completed

### Core Modules (_src/)
- ✅ `image_generator.py` - SD wrapper (txt2img, img2img)
- ✅ `frame_interpolator.py` - Frame interpolation (blend, optical_flow, film)
- ✅ `video_animator.py` - Basic Deforum loop (no interpolation)
- ✅ `_video_animator.py` - Keyframe + interpolation system

### Features
- ✅ Deforum-style animation (txt2img → img2img chain)
- ✅ Frame interpolation with keyframes (2x, 4x, 8x speedup)
- ✅ Alpha blending interpolation


### Testing
- ✅ `test_image_generator.py` - SD generation tests
- ✅ `test_video_animator.py` - Basic video tests
- ✅ `test_interpolation.py` - Interpolation comparison tests
- ✅ Global `TOTAL_FRAMES` variable for easy control

### Utilities
- ✅ `frames_to_video.py` - Convert any frame directory to video

---

## 🚧 In Progress / Next Steps

### Interpolation Enhancements
- ⏳ Add RIFE neural interpolation (optional dependency)

### Beat Integration
- ⏳ Connect fixed beat schedule to video animator
- ⏳ Adaptive keyframing (more keyframes on beats)
- ⏳ Dynamic strength based on beats

### Additional Modules (from refactoring plan)
- ⏳ Frame transformer (zoom, rotate, translate) --> look at video_effects.txt
- ⏳ Audio analyzer refactor
- ⏳ Music schedule refactor
- ⏳ Complete integration of all modules

---

## 📋 Future Ideas

- Prompt interpolation (change prompts over time)
- Quality comparison mode (test all interpolation methods)
- Batch processing support
- Real-time preview mode
- Web UI for parameter control
