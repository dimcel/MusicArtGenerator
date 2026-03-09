"""
Video Animator - Creates animations using Deforum-style frame generation

Implements the Deforum animation loop:
1. Frame 0: prompt + noise → diffusion → image
2. Frame N: previous_image + noise → diffusion → new_image

Simple, focused module that orchestrates image generation into video.
"""

import os
from pathlib import Path
from PIL import Image
from typing import List, Optional
from dataclasses import dataclass

from src.image_generator import ImageGenerator, ImageGenerationConfig


@dataclass
class VideoAnimationConfig:
    """Configuration for video animation"""
    # Frame settings
    total_frames: int = 60
    fps: int = 24
    
    # Generation settings
    prompt: str = "old man sitting on a bench, thinking, peaceful park"
    negative_prompt: str = "blurry, bad quality, distorted"
    strength: float = 0.6  # How much to change each frame (0.0-1.0)
    
    # Output settings
    output_dir: str = "animation_output"
    width: int = 512
    height: int = 512
    
    # Seed for reproducibility
    seed: int = 42


class VideoAnimator:
    """
    Creates video animations using Deforum-style frame-by-frame generation.
    
    Process:
    1. Generate first frame from text prompt
    2. For each subsequent frame:
       - Take previous frame
       - Add noise (controlled by strength)
       - Run diffusion with prompt
       - Get new frame
    3. Save all frames
    4. Create video from frames
    
    Usage:
        config = VideoAnimationConfig(
            total_frames=60,
            prompt="old man on bench",
            strength=0.6
        )
        
        animator = VideoAnimator(config)
        animator.generate_animation()
    """
    
    def __init__(self, config: VideoAnimationConfig):
        """
        Initialize the animator.
        
        Args:
            config: Animation configuration
        """
        self.config = config
        
        # Create image generator with matching settings
        img_config = ImageGenerationConfig(
            width=config.width,
            height=config.height,
            negative_prompt=config.negative_prompt
        )
        self.generator = ImageGenerator(img_config)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Video Animator")
        print(f"   Frames: {config.total_frames}")
        print(f"   FPS: {config.fps}")
        print(f"   Duration: {config.total_frames / config.fps:.1f}s")
        print(f"   Strength: {config.strength}")
        print(f"   Output: {config.output_dir}/")
    
    def generate_animation(self) -> List[str]:
        """
        Generate the complete animation.
        
        Returns:
            List of frame file paths
        """
        print(f"\n{'='*70}")
        print("GENERATING ANIMATION")
        print(f"{'='*70}")
        
        frame_paths = []
        current_image = None
        
        for frame_num in range(self.config.total_frames):
            print(f"\n[{frame_num + 1}/{self.config.total_frames}] ", end="")
            
            if frame_num == 0:
                # First frame: Generate from text
                print(f"txt2img → '{self.config.prompt[:40]}...'")
                current_image = self.generator.generate_from_text(
                    prompt=self.config.prompt,
                    seed=self.config.seed
                )
            else:
                # Subsequent frames: Generate from previous image
                print(f"img2img (strength={self.config.strength})")
                current_image = self.generator.generate_from_image(
                    init_image=current_image,
                    prompt=self.config.prompt,
                    strength=self.config.strength,
                    seed=self.config.seed + frame_num  # Vary seed slightly
                )
            
            # Save frame
            frame_path = os.path.join(
                self.config.output_dir,
                f"frame_{frame_num:05d}.png"
            )
            current_image.save(frame_path)
            frame_paths.append(frame_path)
            
            print(f"   ✓ Saved: {frame_path}")
        
        print(f"\n{'='*70}")
        print(f"✓ Generated {len(frame_paths)} frames")
        print(f"{'='*70}")
        
        return frame_paths
    # reminder, this is likely better to be in utils
    def create_video(
        self,
        frame_paths: List[str] = None,
        output_path: str = None
    ) -> str:
        """
        Create video from frames using ffmpeg.
        
        Args:
            frame_paths: List of frame paths (uses all frames in output_dir if None)
            output_path: Output video path (default: output_dir/animation.mp4)
            
        Returns:
            Path to created video
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "animation.mp4")
        
        # Build ffmpeg command
        frame_pattern = os.path.join(self.config.output_dir, "frame_%05d.png")
        
        cmd = (
            f'ffmpeg -y -framerate {self.config.fps} '
            f'-i "{frame_pattern}" '
            f'-c:v libx264 -pix_fmt yuv420p '
            f'-crf 23 '
            f'"{output_path}"'
        )
        
        print(f"\n🎬 Creating video...")
        print(f"   Command: {cmd[:80]}...")
        
        result = os.system(cmd)
        
        if result == 0:
            print(f"   ✓ Video created: {output_path}")
            return output_path
        else:
            print(f"   ❌ Failed to create video (ffmpeg error)")
            return None
    
    def generate_and_save_video(self) -> str:
        """
        Complete pipeline: Generate frames and create video.
        
        Returns:
            Path to created video
        """
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
            print(f"{'='*70}")
        
        return video_path

# this is a test, it should be removed, or go to test folder
def simple_animation(
    prompt: str,
    frames: int = 60,
    strength: float = 0.6,
    output_dir: str = "animation_output"
) -> str:
    """
    Quick helper function to generate a simple animation.
    
    Args:
        prompt: Text description
        frames: Number of frames
        strength: Change amount per frame
        output_dir: Where to save
        
    Returns:
        Path to video file
        
    Example:
        video = simple_animation(
            prompt="old man on bench, autumn leaves",
            frames=60,
            strength=0.6
        )
    """
    config = VideoAnimationConfig(
        prompt=prompt,
        total_frames=frames,
        strength=strength,
        output_dir=output_dir
    )
    
    animator = VideoAnimator(config)
    return animator.generate_and_save_video()
