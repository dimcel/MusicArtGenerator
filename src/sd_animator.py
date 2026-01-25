"""
Stable Diffusion Animation Generator
Integrates with diffusers library to create Deforum-style animations
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, List
import os
from pathlib import Path

# Import our interpolation engine
from src.interpolation_engine import AnimationInterpolator, Keyframe, TransformationEngine

from src.audio_analyzer import AudioAnalyzer
from src.music_schedule import MusicSchedule, MusicSyncConfig, apply_music_to_schedule


from src.optical_flow_cadence import (
    CadenceConfig, 
    CadenceInterpolator, 
    OpticalFlowMethod,
    ColorCoherenceMethod,
    pil_to_numpy,
    numpy_to_pil
)

class StableDiffusionAnimator:
    """
    Creates animations using Stable Diffusion with frame-to-frame coherence
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the animator with a Stable Diffusion model
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_id = model_id
        self.pipe = None
        self.transformer = TransformationEngine()
        
        print(f"Initializing on {device}...")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline"""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                # Optionally enable xformers if available
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
            
            print("✓ Pipeline loaded successfully")
            
        except ImportError:
            print("ERROR: diffusers library not installed")
            print("Install with: pip install diffusers transformers accelerate --break-system-packages")
            raise
    
    def generate_first_frame(self, prompt: str, negative_prompt: str = "",
                           width: int = 512, height: int = 512,
                           guidance_scale: float = 7.5,
                           num_inference_steps: int = 50,
                           seed: int = -1) -> Image.Image:
        """
        Generate the initial frame using txt2img
        """
        from diffusers import StableDiffusionPipeline
        
        # For first frame, we need txt2img
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        txt2img_pipe = txt2img_pipe.to(self.device)
        
        if self.device == "cuda":
            txt2img_pipe.enable_attention_slicing()
        
        # Set seed if specified
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"Generating first frame: '{prompt[:50]}...'")
        
        image = txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        # Clean up txt2img pipeline
        del txt2img_pipe
        torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return image
    
    def generate_frame(self, init_image: Image.Image, prompt: str,
                      negative_prompt: str = "", strength: float = 0.75,
                      guidance_scale: float = 7.5,
                      num_inference_steps: int = 50,
                      seed: int = -1) -> Image.Image:
        """
        Generate a single frame using img2img
        
        Args:
            init_image: Previous frame to use as initialization
            prompt: Text prompt for this frame
            strength: Denoising strength (0-1, higher = more change)
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            seed: Random seed (-1 for random)
        """
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        return image
    
    def apply_camera_motion(self, image: Image.Image, zoom: float,
                          angle: float, tx: float, ty: float) -> Image.Image:
        """
        Apply camera transformations to image
        """
        img_array = np.array(image).astype(np.float32)
        transformed = self.transformer.apply_transforms(img_array, zoom, angle, tx, ty)
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        return Image.fromarray(transformed)
    
    def render_animation(self, interpolator: AnimationInterpolator,
                        output_dir: str = "output_frames",
                        width: int = 512, height: int = 512,
                        num_inference_steps: int = 30,
                        save_every_nth: int = 1) -> List[str]:
        """
        Render the complete animation
        
        Args:
            interpolator: AnimationInterpolator with keyframes
            output_dir: Directory to save frames
            width: Frame width
            height: Frame height
            num_inference_steps: Denoising steps per frame
            save_every_nth: Save every nth frame (for faster preview)
        
        Returns:
            List of output frame paths
        """
        Path(output_dir).mkdir(exist_ok=True)
        frame_paths = []
        
        # Get schedule
        schedule = interpolator.generate_animation_schedule()
        total_frames = len(schedule)
        
        print(f"\nRendering {total_frames} frames...")
        print("=" * 60)
        
        # Generate first frame
        first_params = schedule[0]
        current_frame = self.generate_first_frame(
            prompt=first_params['prompt'],
            negative_prompt=first_params['negative_prompt'],
            width=width,
            height=height,
            guidance_scale=first_params['guidance_scale'],
            num_inference_steps=num_inference_steps,
            seed=first_params['seed']
        )
        
        # Save first frame
        frame_path = os.path.join(output_dir, f"frame_{0:05d}.png")
        current_frame.save(frame_path)
        frame_paths.append(frame_path)
        print(f"[0/{total_frames}] Saved: {frame_path}")
        
        # Generate subsequent frames
        for i in range(1, total_frames):
            params = schedule[i]
            
            # Apply camera motion to previous frame
            if (params['zoom'] != 1.0 or params['angle'] != 0.0 or 
                params['translation_x'] != 0.0 or params['translation_y'] != 0.0):
                current_frame = self.apply_camera_motion(
                    current_frame,
                    zoom=params['zoom'],
                    angle=params['angle'],
                    tx=params['translation_x'],
                    ty=params['translation_y']
                )
            
            # Generate new frame with img2img
            current_frame = self.generate_frame(
                init_image=current_frame,
                prompt=params['prompt'],
                negative_prompt=params['negative_prompt'],
                strength=params['strength'],
                guidance_scale=params['guidance_scale'],
                num_inference_steps=num_inference_steps,
                seed=params['seed']
            )
            
            # Save frame
            if i % save_every_nth == 0:
                frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
                current_frame.save(frame_path)
                frame_paths.append(frame_path)
                print(f"[{i}/{total_frames}] {params['prompt'][:40]}... | "
                      f"strength:{params['strength']:.2f} zoom:{params['zoom']:.2f}")
        
        print("=" * 60)
        print(f"✓ Animation complete! {len(frame_paths)} frames saved to {output_dir}/")
        
        return frame_paths
    
    def create_video(self, frame_dir: str, output_path: str = "animation.mp4",
                    fps: int = 24):
        """
        Combine frames into video using ffmpeg
        """
        try:
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', f'{frame_dir}/frame_*.png',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Video created: {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e.stderr.decode()}")
        except FileNotFoundError:
            print("ffmpeg not found. Install with: apt-get install ffmpeg")

    def render_animation_with_music(
        self, 
        interpolator,
        audio_path: str,
        music_config=None,
        output_dir: str = "output_frames",
        width: int = 512, 
        height: int = 512,
        num_inference_steps: int = 30,
        save_every_nth: int = 1,
        fps: int = 24
    ):
        """
        Render animation synchronized to music.
        
        Args:
            interpolator: Your AnimationInterpolator with keyframes
            audio_path: Path to music file (mp3, wav, etc.)
            music_config: MusicSyncConfig for how audio affects params
            output_dir: Where to save frames
            width: Frame width
            height: Frame height
            num_inference_steps: Denoising steps per frame
            save_every_nth: Save every nth frame
            fps: Frames per second (for audio sync)
        
        Returns:
            List of output frame paths
        """
        from src.audio_analyzer import AudioAnalyzer
        from src.music_schedule import MusicSchedule, MusicSyncConfig, apply_music_to_schedule
        
        # Default config if none provided
        if music_config is None:
            music_config = MusicSyncConfig(
                amplitude_to_strength=(0.5, 0.75),
                beat_zoom_boost=0.03,
            )
        
        # 1. Analyze audio
        print(f"\n🎵 Analyzing audio: {audio_path}")
        analyzer = AudioAnalyzer(audio_path)
        analyzer.load()
        
        # 2. Create music schedule
        music = MusicSchedule(analyzer, fps=fps, total_frames=interpolator.total_frames)
        print(music.summary())
        
        # 3. Generate base schedule from keyframes
        base_schedule = interpolator.generate_animation_schedule()
        
        # 4. Generate music modifiers
        music_modifiers = music.generate_modifiers(music_config)
        
        # 5. Merge them
        final_schedule = apply_music_to_schedule(base_schedule, music_modifiers)
        
        # 6. Render using the modified schedule
        return self.render_from_schedule(
            final_schedule, 
            output_dir, 
            width, 
            height, 
            num_inference_steps,
            save_every_nth
        )
    def render_from_schedule(
        self,
        schedule,
        output_dir: str = "output_frames",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        save_every_nth: int = 1
    ):
        """
        Render animation from a pre-computed schedule.
        
        Args:
            schedule: List of dicts with per-frame params
            output_dir: Where to save frames
            width: Frame width
            height: Frame height
            num_inference_steps: Denoising steps per frame
            save_every_nth: Save every nth frame
        
        Returns:
            List of output frame paths
        """
        from pathlib import Path
        
        Path(output_dir).mkdir(exist_ok=True)
        frame_paths = []
        total_frames = len(schedule)
        
        print(f"\n🎬 Rendering {total_frames} frames...")
        print("=" * 60)
        
        # Generate first frame
        first_params = schedule[0]
        current_frame = self.generate_first_frame(
            prompt=first_params['prompt'],
            negative_prompt=first_params.get('negative_prompt', ''),
            width=width,
            height=height,
            guidance_scale=first_params.get('guidance_scale', 7.5),
            num_inference_steps=num_inference_steps,
            seed=first_params.get('seed', -1)
        )
        
        # Save first frame
        frame_path = os.path.join(output_dir, f"frame_{0:05d}.png")
        current_frame.save(frame_path)
        frame_paths.append(frame_path)
        print(f"[0/{total_frames}] Saved: {frame_path}")
        
        # Generate subsequent frames
        for i in range(1, total_frames):
            params = schedule[i]
            
            # Beat indicator for logging
            beat_marker = "🥁" if params.get('is_beat', False) else "  "
            
            # Calculate zoom (base + boost)
            zoom = params.get('zoom', 1.0) + params.get('zoom_boost', 0.0)
            
            # Apply camera motion to previous frame
            if (zoom != 1.0 or 
                params.get('angle', 0) != 0.0 or 
                params.get('translation_x', 0) != 0.0 or 
                params.get('translation_y', 0) != 0.0):
                
                current_frame = self.apply_camera_motion(
                    current_frame,
                    zoom=zoom,
                    angle=params.get('angle', 0.0),
                    tx=params.get('translation_x', 0.0),
                    ty=params.get('translation_y', 0.0)
                )
            
            # Calculate strength (base + boost)
            strength = params.get('strength', 0.6)
            if 'strength_boost' in params:
                strength += params['strength_boost']
            
            # Clamp strength to valid range
            strength = max(0.0, min(1.0, strength))
            
            # Generate new frame with img2img
            current_frame = self.generate_frame(
                init_image=current_frame,
                prompt=params['prompt'],
                negative_prompt=params.get('negative_prompt', ''),
                strength=strength,
                guidance_scale=params.get('guidance_scale', 7.5),
                num_inference_steps=num_inference_steps,
                seed=params.get('seed', -1)
            )
            
            # Save frame
            if i % save_every_nth == 0:
                frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
                current_frame.save(frame_path)
                frame_paths.append(frame_path)
                
                # Log progress with music info
                amp = params.get('amplitude', 0)
                print(f"[{i}/{total_frames}] {beat_marker} "
                    f"amp={amp:.2f} str={strength:.2f} zoom={zoom:.3f}")
        
        print("=" * 60)
        print(f"✓ Animation complete! {len(frame_paths)} frames saved to {output_dir}/")
        
        return frame_paths
    
    def render_with_cadence(
        self,
        interpolator,           # Your AnimationInterpolator
        config: 'CadenceConfig' = None,
        output_dir: str = "output_frames",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        fps: int = 24,
        audio_path: str = None,         # Optional: for music sync
        music_config: 'MusicSyncConfig' = None,  # Optional: music config
    ):
        """
        Render animation using optical flow cadence for faster generation.
        
        Instead of diffusing every frame:
            Frame 0 [DIFFUSE] → Frame 1 [DIFFUSE] → Frame 2 [DIFFUSE] → ...
            
        With cadence=4:
            Frame 0 [DIFFUSE] → warp → warp → warp → Frame 4 [DIFFUSE] → ...
            
        This is 4x faster and often looks smoother!
        
        Args:
            interpolator: Your AnimationInterpolator with keyframes
            config: CadenceConfig (cadence, flow method, color coherence)
            output_dir: Where to save frames
            width, height: Frame dimensions
            num_inference_steps: SD steps per diffusion
            fps: Frames per second (for audio sync)
            audio_path: Optional music file for sync
            music_config: Optional MusicSyncConfig
            
        Returns:
            List of frame file paths
        """
        from src.optical_flow_cadence import (
            CadenceConfig, CadenceInterpolator, 
            pil_to_numpy, numpy_to_pil
        )
        from pathlib import Path
        import os
        
        # Default config
        if config is None:
            config = CadenceConfig(
                cadence=4,
                use_optical_flow=True,
                color_coherence=ColorCoherenceMethod.LAB
            )
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate base schedule
        total_frames = interpolator.total_frames
        base_schedule = interpolator.generate_animation_schedule()
        
        # Optionally merge with music
        if audio_path and music_config:
            from src.audio_analyzer import AudioAnalyzer
            from src.music_schedule import MusicSchedule, apply_music_to_schedule
            
            print(f"\n🎵 Analyzing audio: {audio_path}")
            analyzer = AudioAnalyzer(audio_path)
            analyzer.load()
            music = MusicSchedule(analyzer, fps=fps, total_frames=total_frames)
            music_modifiers = music.generate_modifiers(music_config)
            schedule = apply_music_to_schedule(base_schedule, music_modifiers)
            print(music.summary())
        else:
            schedule = base_schedule
        
        # Create cadence interpolator
        cadence_interp = CadenceInterpolator(config)
        diffusion_frames = cadence_interp.get_diffusion_frames(total_frames)
        
        print(f"\n🎬 Cadence Rendering")
        print(f"   Total frames: {total_frames}")
        print(f"   Diffusion frames: {len(diffusion_frames)} (cadence={config.cadence})")
        print(f"   Speedup: ~{config.cadence}x faster")
        print(f"   Optical flow: {config.use_optical_flow}")
        print(f"   Color coherence: {config.color_coherence.value}")
        print("=" * 60)
        
        all_frames = [None] * total_frames
        frame_paths = []
        
        prev_diffused_pil = None
        prev_diffused_np = None
        prev_diffused_idx = None
        
        for diff_idx in diffusion_frames:
            params = schedule[diff_idx]
            beat_marker = "🥁" if params.get('is_beat', False) else "  "
            
            # Get strength (with any music boost)
            strength = params.get('strength', 0.6)
            if 'strength_boost' in params:
                strength = min(1.0, strength + params['strength_boost'])
            
            # Get zoom (with any music boost)  
            zoom = params.get('zoom', 1.0) + params.get('zoom_boost', 0.0)
            
            print(f"\n[{diff_idx}/{total_frames}] {beat_marker} 🎨 DIFFUSING... "
                f"(str={strength:.2f}, zoom={zoom:.2f})")
            
            if diff_idx == 0:
                # First frame: txt2img
                current_pil = self.generate_first_frame(
                    prompt=params['prompt'],
                    negative_prompt=params.get('negative_prompt', ''),
                    width=width,
                    height=height,
                    guidance_scale=params.get('guidance_scale', 7.5),
                    num_inference_steps=num_inference_steps,
                    seed=params.get('seed', -1)
                )
            else:
                # Apply camera motion to previous diffused frame
                motion_frame = prev_diffused_pil
                if zoom != 1.0 or params.get('angle', 0) != 0:
                    motion_frame = self.apply_camera_motion(
                        motion_frame,
                        zoom=zoom,
                        angle=params.get('angle', 0.0),
                        tx=params.get('translation_x', 0.0),
                        ty=params.get('translation_y', 0.0)
                    )
                
                # img2img diffusion
                current_pil = self.generate_frame(
                    init_image=motion_frame,
                    prompt=params['prompt'],
                    negative_prompt=params.get('negative_prompt', ''),
                    strength=strength,
                    guidance_scale=params.get('guidance_scale', 7.5),
                    num_inference_steps=num_inference_steps,
                    seed=params.get('seed', -1)
                )
            
            current_np = pil_to_numpy(current_pil)
            all_frames[diff_idx] = current_np
            
            # Generate intermediate frames via optical flow
            if prev_diffused_np is not None and prev_diffused_idx is not None:
                num_intermediates = diff_idx - prev_diffused_idx - 1
                if num_intermediates > 0:
                    print(f"    ↳ Interpolating {num_intermediates} frames with optical flow...")
                    
                    intermediates = cadence_interp.generate_cadence_frames(
                        prev_diffused_np,
                        current_np,
                        num_intermediates
                    )
                    
                    for i, frame_np in enumerate(intermediates):
                        frame_idx = prev_diffused_idx + 1 + i
                        all_frames[frame_idx] = frame_np
            
            prev_diffused_pil = current_pil
            prev_diffused_np = current_np
            prev_diffused_idx = diff_idx
        
        # Handle trailing frames (after last diffusion)
        if prev_diffused_idx is not None and prev_diffused_idx < total_frames - 1:
            for i in range(prev_diffused_idx + 1, total_frames):
                all_frames[i] = prev_diffused_np.copy()
        
        # Save all frames
        print(f"\n💾 Saving {total_frames} frames...")
        for i, frame_np in enumerate(all_frames):
            if frame_np is not None:
                frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
                numpy_to_pil(frame_np).save(frame_path)
                frame_paths.append(frame_path)
        
        print("=" * 60)
        print(f"✓ Animation complete!")
        print(f"  Frames saved: {len(frame_paths)}")
        print(f"  Diffusions: {len(diffusion_frames)} (saved {total_frames - len(diffusion_frames)} diffusions!)")
        print(f"  Output: {output_dir}/")
        
        return frame_paths


def example_simple_animation():
    """
    Simple example: zoom in animation with prompt transition
    """
    print("\n" + "=" * 60)
    print("SIMPLE ANIMATION EXAMPLE")
    print("=" * 60)
    
    # Define simple keyframes
    keyframes = [
        Keyframe(
            frame=0,
            prompt="a serene lake at dawn, mist rising, photorealistic",
            strength=0.5,
            zoom=1.0,
            seed=42
        ),
        Keyframe(
            frame=15,
            prompt="a serene lake at dawn with golden sunlight, mist rising, photorealistic",
            strength=0.6,
            zoom=1.15,
        ),
        Keyframe(
            frame=30,
            prompt="a serene lake in morning light, birds flying, photorealistic",
            strength=0.65,
            zoom=1.0,
        ),
    ]
    
    # Create interpolator
    interpolator = AnimationInterpolator(keyframes, total_frames=30)
    
    # Create animator and render
    # NOTE: This requires GPU and diffusers library
    # animator = StableDiffusionAnimator()
    # animator.render_animation(interpolator, num_inference_steps=20)
    
    # For now, just show the schedule
    print("\nAnimation schedule (every 5th frame):")
    for i in range(0, 30, 5):
        params = interpolator.get_frame_params(i)
        print(f"\nFrame {i:2d}:")
        print(f"  Prompt: {params['prompt'][:50]}...")
        print(f"  Strength: {params['strength']:.3f}, Zoom: {params['zoom']:.3f}")
    
    return interpolator


if __name__ == "__main__":
    print("Stable Diffusion Animation Generator\n")
    
    # Run example
    interpolator = example_simple_animation()
    
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("""
To actually generate frames, you need:

1. Install dependencies:
   pip install diffusers transformers accelerate --break-system-packages
   pip install torch torchvision --break-system-packages

2. Run the animation:
   
   from sd_animator import StableDiffusionAnimator, Keyframe, AnimationInterpolator
   
   keyframes = [...]  # Define your keyframes
   interpolator = AnimationInterpolator(keyframes, total_frames=60)
   
   animator = StableDiffusionAnimator()
   animator.render_animation(interpolator, output_dir="my_animation")
   animator.create_video("my_animation", "output.mp4", fps=24)

3. For faster testing, reduce:
   - num_inference_steps (try 15-20)
   - frame resolution (try 512x512)
   - total_frames (try 20-30 for tests)

4. Key parameters:
   - strength: Higher = more change per frame (0.5-0.8 typical)
   - zoom: Smooth camera zoom (1.0 = no zoom)
   - angle: Rotation in degrees
   - translation_x/y: Pan camera in pixels
    """)
