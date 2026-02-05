"""
Video Animator with Frame Interpolation

Creates animations using keyframe generation + interpolation:
1. Generate keyframes with diffusion (expensive)
2. Interpolate in-between frames (cheap)
3. Result: Faster generation with smooth video

Example:
    Without interpolation: Generate 60 frames = 5 minutes
    With 1:4 interpolation: Generate 15 keyframes + interpolate 45 = 1.5 minutes
"""

import os
from pathlib import Path
from PIL import Image
from typing import List, Optional
from dataclasses import dataclass

from _src.image_generator import ImageGenerator, ImageGenerationConfig
from _src.frame_interpolator import FrameInterpolator


@dataclass
class VideoAnimationConfig:
    """Configuration for video animation with interpolation"""
    # Frame settings
    total_frames: int = 60
    fps: int = 24
    
    # Interpolation settings
    use_interpolation: bool = True  # Enable/disable interpolation
    keyframe_interval: int = 4      # Generate every Nth frame, interpolate rest
    interpolation_method: str = "blend"  # Interpolation method: blend, optical_flow, rife, film
    
    # Generation settings
    prompt: str = "old man sitting on a bench, thinking, peaceful park"
    negative_prompt: str = "blurry, bad quality, distorted"
    strength: float = 0.6  # How much to change each keyframe
    
    # Output settings
    output_dir: str = "animation_output"
    width: int = 512
    height: int = 512
    
    # Seed for reproducibility
    seed: int = 42


class VideoAnimator:
    """
    Creates video animations with optional frame interpolation.
    
    Process without interpolation (use_interpolation=False):
    1. Generate all frames with diffusion
    2. Save frames
    3. Create video
    
    Process with interpolation (use_interpolation=True):
    1. Generate keyframes only (every Nth frame)
    2. Interpolate in-between frames
    3. Save all frames
    4. Create video
    
    Usage:
        # Fast mode (4x faster)
        config = VideoAnimationConfig(
            use_interpolation=True,
            keyframe_interval=4  # Generate every 4th frame
        )
        
        # Quality mode (slower but better)
        config = VideoAnimationConfig(
            use_interpolation=False  # Generate all frames
        )
        
        animator = VideoAnimator(config)
        animator.generate_and_save_video()
    """
    
    def __init__(self, config: VideoAnimationConfig):
        """Initialize the animator."""
        self.config = config
        
        # Create image generator
        img_config = ImageGenerationConfig(
            width=config.width,
            height=config.height,
            negative_prompt=config.negative_prompt
        )
        self.generator = ImageGenerator(img_config)
        
        # Create frame interpolator with specified method
        self.interpolator = FrameInterpolator(method=config.interpolation_method)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate stats
        if config.use_interpolation:
            num_keyframes = (config.total_frames + config.keyframe_interval - 1) // config.keyframe_interval
            num_interpolated = config.total_frames - num_keyframes
            speedup = config.total_frames / num_keyframes
        else:
            num_keyframes = config.total_frames
            num_interpolated = 0
            speedup = 1.0
        
        print(f"\n🎬 Video Animator")
        print(f"   Total frames: {config.total_frames}")
        print(f"   FPS: {config.fps}")
        print(f"   Duration: {config.total_frames / config.fps:.1f}s")
        print(f"   Strength: {config.strength}")
        
        if config.use_interpolation:
            print(f"\n   🚀 Interpolation: ENABLED")
            print(f"      Method: {config.interpolation_method}")
            print(f"      Keyframe interval: every {config.keyframe_interval} frames")
            print(f"      Generated: {num_keyframes} frames")
            print(f"      Interpolated: {num_interpolated} frames")
            print(f"      Speedup: ~{speedup:.1f}x faster")
        else:
            print(f"\n   🐢 Interpolation: DISABLED")
            print(f"      Generating all {num_keyframes} frames")
        
        print(f"\n   Output: {config.output_dir}/")
    
    def _is_keyframe(self, frame_num: int) -> bool:
        """Check if frame should be generated (vs interpolated)"""
        if not self.config.use_interpolation:
            return True  # All frames are keyframes
        
        # Frame 0 is always a keyframe
        if frame_num == 0:
            return True
        
        # Every Nth frame is a keyframe
        return frame_num % self.config.keyframe_interval == 0
    
    def generate_animation(self) -> List[str]:
        """
        Generate the complete animation with optional interpolation.
        
        Two-pass approach:
        1. Generate all keyframes with diffusion
        2. Interpolate in-between frames
        
        Returns:
            List of frame file paths
        """
        print(f"\n{'='*70}")
        print("GENERATING ANIMATION")
        print(f"{'='*70}")
        
        # PASS 1: Generate keyframes
        print(f"\n📍 PASS 1: Generating Keyframes")
        print(f"{'='*70}")
        
        keyframes = {}  # frame_num -> PIL Image
        keyframe_nums = [i for i in range(self.config.total_frames) if self._is_keyframe(i)]
        current_image = None
        
        for idx, frame_num in enumerate(keyframe_nums, 1):
            print(f"\n[Keyframe {idx}/{len(keyframe_nums)}] Frame {frame_num}: ", end="")
            
            if frame_num == 0:
                print(f"txt2img → '{self.config.prompt[:40]}...'")
                current_image = self.generator.generate_from_text(
                    prompt=self.config.prompt,
                    seed=self.config.seed
                )
            else:
                print(f"img2img (strength={self.config.strength})")
                current_image = self.generator.generate_from_image(
                    init_image=current_image,
                    prompt=self.config.prompt,
                    strength=self.config.strength,
                    seed=self.config.seed + frame_num
                )
            
            # Save keyframe
            frame_path = os.path.join(
                self.config.output_dir,
                f"frame_{frame_num:05d}.png"
            )
            current_image.save(frame_path)
            keyframes[frame_num] = current_image
            
            print(f"   ✓ Saved: {frame_path}")
        
        print(f"\n{'='*70}")
        print(f"✓ Generated {len(keyframes)} keyframes")
        print(f"{'='*70}")
        
        # PASS 2: Interpolate (if enabled)
        if self.config.use_interpolation and len(keyframes) > 1:
            print(f"\n🔄 PASS 2: Interpolating Frames")
            print(f"{'='*70}")
            
            interpolated_count = 0
            keyframe_list = sorted(keyframes.keys())
            
            for i in range(len(keyframe_list) - 1):
                frame_a_num = keyframe_list[i]
                frame_b_num = keyframe_list[i + 1]
                frame_a = keyframes[frame_a_num]
                frame_b = keyframes[frame_b_num]
                
                # Calculate frames between
                frames_between = frame_b_num - frame_a_num - 1
                
                if frames_between > 0:
                    print(f"\n  Interpolating {frames_between} frames between {frame_a_num} and {frame_b_num}")
                    
                    # Generate interpolated frames
                    interpolated_frames = self.interpolator.interpolate(
                        frame_a, frame_b, num_frames=frames_between
                    )
                    
                    # Save interpolated frames
                    for j, interp_frame in enumerate(interpolated_frames):
                        frame_num = frame_a_num + j + 1
                        frame_path = os.path.join(
                            self.config.output_dir,
                            f"frame_{frame_num:05d}.png"
                        )
                        interp_frame.save(frame_path)
                        interpolated_count += 1
                        
                        if (j + 1) % 10 == 0 or (j + 1) == len(interpolated_frames):
                            print(f"    ✓ Saved {j + 1}/{frames_between} frames")
            
            print(f"\n{'='*70}")
            print(f"✓ Interpolated {interpolated_count} frames")
            print(f"{'='*70}")
        
        # Collect all frame paths
        frame_paths = []
        for frame_num in range(self.config.total_frames):
            frame_path = os.path.join(
                self.config.output_dir,
                f"frame_{frame_num:05d}.png"
            )
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
        
        print(f"\n{'='*70}")
        print(f"✓ TOTAL: {len(frame_paths)} frames saved")
        print(f"{'='*70}")
        
        return frame_paths
    
    def create_video(
        self,
        frame_paths: List[str] = None,
        output_path: str = None
    ) -> str:
        """Create video from frames using ffmpeg (handles gaps)."""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "animation.mp4")
        
        # If frame_paths not provided, find all frames
        if frame_paths is None:
            from pathlib import Path
            frames_dir = Path(self.config.output_dir)
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
        
        if not frame_paths:
            print(f"   ❌ No frames found to create video")
            return None
        
        print(f"\n🎬 Creating video...")
        print(f"   Frames: {len(frame_paths)}")
        print(f"   FPS: {self.config.fps}")
        print(f"   Output: {output_path}")
        
        # Create temporary file list for ffmpeg concat demuxer
        # This handles gaps in frame sequences
        from pathlib import Path
        filelist_path = Path(self.config.output_dir) / "ffmpeg_filelist.txt"
        
        try:
            with open(filelist_path, 'w') as f:
                for frame_path in frame_paths:
                    # Write absolute path with proper escaping
                    abs_path = os.path.abspath(frame_path)
                    f.write(f"file '{abs_path}'\n")
            
            # ffmpeg command using concat demuxer
            cmd = (
                f'ffmpeg -y -f concat -safe 0 -r {self.config.fps} '
                f'-i "{filelist_path}" '
                f'-c:v libx264 -pix_fmt yuv420p '
                f'-crf 23 '
                f'"{output_path}"'
            )
            
            print(f"   Running ffmpeg...")
            result = os.system(cmd)
            
            # Clean up temporary file
            filelist_path.unlink()
            
            if result == 0:
                duration = len(frame_paths) / self.config.fps
                print(f"   ✓ Video created: {output_path}")
                print(f"   Duration: {duration:.2f} seconds")
                return output_path
            else:
                print(f"   ❌ Failed to create video (ffmpeg error)")
                return None
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if filelist_path.exists():
                filelist_path.unlink()
            return None
    
    def generate_and_save_video(self) -> str:
        """Complete pipeline: Generate frames and create video."""
        # Generate frames
        frame_paths = self.generate_animation()
        
        # Create video
        video_path = self.create_video(frame_paths)
        
        if video_path:
            print(f"\n{'='*70}")
            print("✓ ANIMATION COMPLETE!")
            print(f"{'='*70}")
            print(f"  Frames: {self.config.output_dir}/frame_*.png")
            print(f"  Video:  {video_path}")
            print(f"  Duration: {self.config.total_frames / self.config.fps:.1f}s")
            
            if self.config.use_interpolation:
                num_keyframes = sum(1 for i in range(self.config.total_frames) if self._is_keyframe(i))
                print(f"  Keyframes generated: {num_keyframes}")
                print(f"  Frames interpolated: {self.config.total_frames - num_keyframes}")
            
            print(f"{'='*70}")
        
        return video_path


def simple_animation(
    prompt: str,
    frames: int = 60,
    strength: float = 0.6,
    use_interpolation: bool = True,
    keyframe_interval: int = 4,
    interpolation_method: str = "blend",
    output_dir: str = "animation_output"
) -> str:
    """
    Quick helper function to generate animation with interpolation.
    
    Args:
        prompt: Text description
        frames: Number of frames
        strength: Change amount per keyframe
        use_interpolation: Enable frame interpolation
        keyframe_interval: Generate every Nth frame
        interpolation_method: Method to use (blend, optical_flow, rife, film)
        output_dir: Where to save
        
    Returns:
        Path to video file
        
    Example:
        # Fast mode with optical flow
        video = simple_animation(
            prompt="old man on bench",
            frames=60,
            use_interpolation=True,
            keyframe_interval=4,
            interpolation_method="optical_flow"
        )
        
        # Quality mode (slower)
        video = simple_animation(
            prompt="old man on bench",
            frames=60,
            use_interpolation=False
        )
    """
    config = VideoAnimationConfig(
        prompt=prompt,
        total_frames=frames,
        strength=strength,
        use_interpolation=use_interpolation,
        keyframe_interval=keyframe_interval,
        interpolation_method=interpolation_method,
        output_dir=output_dir
    )
    
    animator = VideoAnimator(config)
    return animator.generate_and_save_video()


# =============================================================================
# FUTURE ENHANCEMENTS
# =============================================================================
#
# 1. ADAPTIVE KEYFRAMING
# ----------------------
# - More keyframes during beats (when strength changes)
# - Fewer keyframes during stable sections
# - Example:
#     def _is_keyframe(self, frame_num: int) -> bool:
#         if self.beat_schedule.is_beat(frame_num):
#             return True  # Always generate on beats
#         return frame_num % self.config.keyframe_interval == 0
#
# 2. BETTER INTERPOLATION METHODS
# --------------------------------
# - Upgrade FrameInterpolator to use RIFE or FILM
# - Add quality comparison mode
# - Allow per-frame interpolation method selection
#
# 3. PROMPT INTERPOLATION
# ------------------------
# - Allow prompts to change over time
# - Interpolate prompts between keyframes
# - Example: "old man" → "young man" over 60 frames
#
# 4. DYNAMIC STRENGTH
# -------------------
# - Vary strength per keyframe based on beats
# - High strength on beats, low strength between
# - Smoother overall animation
#
# =============================================================================
