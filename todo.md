# Project TODO - Music Art Generator Refactoring

## ✅ Completed

### Core Modules (_src/)
- ✅ `image_generator.py` - SD wrapper (txt2img, img2img)
- ✅ `frame_interpolator.py` - Frame interpolation (blend, optical_flow, film)
- ✅ `video_animator.py` - Basic Deforum loop (no interpolation)
- ✅ `_video_animator.py` - Keyframe + interpolation system

### Features
- ✅ Fixed beat schedule (1 beat per N seconds)
- ✅ Disabled amplitude modulation (stopped rapid changes)
- ✅ Deforum-style animation (txt2img → img2img chain)
- ✅ Frame interpolation with keyframes (2x, 4x, 8x speedup)
- ✅ Alpha blending interpolation
- ✅ OpenCV optical flow interpolation
- ✅ Concat demuxer for gap-free video creation

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
- ✅ Alpha blending interpolation (simple, fast)
- ✅ OpenCV optical flow interpolation (motion-aware)
- ✅ FILM neural interpolation (best quality, needs TensorFlow)
- ⏳ Fix FILM tensor shape issue
- ⏳ Add RIFE neural interpolation (optional dependency)
- ⏳ Test and compare all interpolation methods

### Beat Integration
- ⏳ Connect fixed beat schedule to video animator
- ⏳ Adaptive keyframing (more keyframes on beats)
- ⏳ Dynamic strength based on beats

### Additional Modules (from refactoring plan)
- ⏳ Frame transformer (zoom, rotate, translate)
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
