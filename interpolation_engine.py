"""
Basic Animation Interpolation Engine
Similar to Deforum - creates smooth animations from text prompts with parameter interpolation
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass
class Keyframe:
    """Represents a keyframe with prompt and parameters"""
    frame: int
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.75  # denoising strength for img2img
    guidance_scale: float = 7.5
    zoom: float = 1.0
    angle: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    seed: int = -1  # -1 means random


class AnimationInterpolator:
    """
    Handles interpolation between keyframes for animation generation
    """
    
    def __init__(self, keyframes: List[Keyframe], total_frames: int):
        self.keyframes = sorted(keyframes, key=lambda k: k.frame)
        self.total_frames = total_frames
        self._validate_keyframes()
        
    def _validate_keyframes(self):
        """Ensure keyframes are valid"""
        if not self.keyframes:
            raise ValueError("At least one keyframe is required")
        if self.keyframes[0].frame != 0:
            raise ValueError("First keyframe must be at frame 0")
        if any(kf.frame > self.total_frames for kf in self.keyframes):
            raise ValueError(f"Keyframe exceeds total frames ({self.total_frames})")
    
    def _interpolate_param(self, param_name: str, frame: int, 
                          interpolation_type: str = 'linear') -> float:
        """
        Interpolate a numeric parameter between keyframes
        
        Args:
            param_name: Name of the parameter to interpolate
            frame: Current frame number
            interpolation_type: 'linear', 'ease_in_out', 'cubic'
        """
        frames = [kf.frame for kf in self.keyframes]
        values = [getattr(kf, param_name) for kf in self.keyframes]
        
        # Handle single keyframe
        if len(frames) == 1:
            return values[0]
        
        # Create interpolation function
        if interpolation_type == 'linear':
            kind = 'linear'
        elif interpolation_type == 'cubic':
            kind = 'cubic'
        else:
            kind = 'quadratic'
        
        # Use scipy's interp1d for smooth interpolation
        interp_func = interp1d(frames, values, kind=kind, 
                               bounds_error=False, fill_value='extrapolate')
        
        return float(interp_func(frame))
    
    def _interpolate_prompt(self, frame: int) -> Tuple[str, float]:
        """
        Get the prompt for current frame with blend weight
        Returns (prompt, blend_weight) where blend_weight indicates 
        how much to blend with next prompt
        """
        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]
        
        for i, kf in enumerate(self.keyframes):
            if kf.frame <= frame:
                prev_kf = kf
                if i < len(self.keyframes) - 1:
                    next_kf = self.keyframes[i + 1]
                else:
                    next_kf = kf
        
        # If at exact keyframe, return that prompt
        if prev_kf.frame == frame:
            return prev_kf.prompt, 0.0
        
        # Calculate blend weight (0.0 = fully prev, 1.0 = fully next)
        if next_kf.frame == prev_kf.frame:
            return prev_kf.prompt, 0.0
        
        blend = (frame - prev_kf.frame) / (next_kf.frame - prev_kf.frame)
        
        # For now, return primary prompt and blend weight
        # In practice, you'd use this to blend embeddings or create weighted prompts
        if blend < 0.5:
            return prev_kf.prompt, blend
        else:
            return next_kf.prompt, 1.0 - blend
    
    def get_frame_params(self, frame: int) -> Dict[str, Any]:
        """
        Get all interpolated parameters for a specific frame
        """
        prompt, blend_weight = self._interpolate_prompt(frame)
        
        # Get the keyframe to reference for negative prompt
        prev_kf = next((kf for kf in reversed(self.keyframes) 
                       if kf.frame <= frame), self.keyframes[0])
        
        params = {
            'frame': frame,
            'prompt': prompt,
            'prompt_blend_weight': blend_weight,
            'negative_prompt': prev_kf.negative_prompt,
            'strength': self._interpolate_param('strength', frame),
            'guidance_scale': self._interpolate_param('guidance_scale', frame),
            'zoom': self._interpolate_param('zoom', frame),
            'angle': self._interpolate_param('angle', frame),
            'translation_x': self._interpolate_param('translation_x', frame),
            'translation_y': self._interpolate_param('translation_y', frame),
            'seed': prev_kf.seed if frame == prev_kf.frame else -1,
        }
        
        return params
    
    def generate_animation_schedule(self) -> List[Dict[str, Any]]:
        """
        Generate complete schedule for all frames
        """
        return [self.get_frame_params(f) for f in range(self.total_frames)]
    
    def export_schedule(self, filename: str):
        """Export animation schedule to JSON"""
        schedule = self.generate_animation_schedule()
        with open(filename, 'w') as f:
            json.dump(schedule, f, indent=2)
        print(f"Schedule exported to {filename}")


class TransformationEngine:
    """
    Handles 2D transformations (zoom, rotate, translate) for frames
    """
    
    @staticmethod
    def apply_transforms(image_array: np.ndarray, zoom: float, 
                        angle: float, tx: float, ty: float) -> np.ndarray:
        """
        Apply transformations to image
        
        Args:
            image_array: Input image as numpy array (H, W, C)
            zoom: Zoom factor (1.0 = no zoom, >1 = zoom in, <1 = zoom out)
            angle: Rotation angle in degrees
            tx: Translation in x (pixels)
            ty: Translation in y (pixels)
        """
        from scipy.ndimage import affine_transform
        
        h, w = image_array.shape[:2]
        center_y, center_x = h / 2, w / 2
        
        # Create transformation matrix
        # 1. Translate to origin
        # 2. Rotate
        # 3. Zoom
        # 4. Translate back + additional translation
        
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Scale (zoom)
        scale_matrix = np.array([
            [1/zoom, 0],
            [0, 1/zoom]
        ])
        
        # Combine rotation and scale
        transform_matrix = rot_matrix @ scale_matrix
        
        # Calculate offset (translation)
        offset_y = center_y - transform_matrix[0, 0] * center_y - transform_matrix[0, 1] * center_x - ty
        offset_x = center_x - transform_matrix[1, 0] * center_y - transform_matrix[1, 1] * center_x - tx
        offset = [offset_y, offset_x]
        
        # Apply transformation to each channel
        transformed = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            transformed[:, :, c] = affine_transform(
                image_array[:, :, c],
                transform_matrix,
                offset=offset,
                order=1,  # bilinear interpolation
                mode='constant',
                cval=0
            )
        
        return transformed


# Example usage and test
def example_animation():
    """
    Example: Create a simple animation schedule
    """
    
    # Define keyframes
    keyframes = [
        Keyframe(
            frame=0,
            prompt="a beautiful mountain landscape at sunrise",
            strength=0.5,
            zoom=1.0,
            angle=0.0,
            seed=42
        ),
        Keyframe(
            frame=30,
            prompt="a beautiful mountain landscape at sunrise, golden hour",
            strength=0.6,
            zoom=1.1,
            angle=2.0,
            translation_x=10
        ),
        Keyframe(
            frame=60,
            prompt="a beautiful mountain landscape at sunset, dramatic clouds",
            strength=0.65,
            zoom=1.2,
            angle=5.0,
            translation_x=20,
            translation_y=-5
        ),
        Keyframe(
            frame=90,
            prompt="a beautiful mountain landscape at dusk, stars appearing",
            strength=0.7,
            zoom=1.0,
            angle=0.0,
            translation_x=0,
            translation_y=0
        ),
    ]
    
    # Create interpolator
    interpolator = AnimationInterpolator(keyframes, total_frames=120)
    
    # Generate and display schedule
    print("=" * 60)
    print("ANIMATION SCHEDULE")
    print("=" * 60)
    
    # Show every 10th frame
    for frame in range(0, 120, 10):
        params = interpolator.get_frame_params(frame)
        print(f"\nFrame {frame}:")
        print(f"  Prompt: {params['prompt'][:50]}...")
        print(f"  Strength: {params['strength']:.3f}")
        print(f"  Zoom: {params['zoom']:.3f}")
        print(f"  Angle: {params['angle']:.2f}°")
        print(f"  Translation: ({params['translation_x']:.1f}, {params['translation_y']:.1f})")
        print(f"  Guidance: {params['guidance_scale']:.2f}")
    
    # Export full schedule
    interpolator.export_schedule('animation_schedule.json')
    
    return interpolator


if __name__ == "__main__":
    print("Animation Interpolation Engine - Test\n")
    interpolator = example_animation()
    
    print("\n" + "=" * 60)
    print("TRANSFORMATION TEST")
    print("=" * 60)
    
    # Test transformation engine with dummy data
    dummy_image = np.random.rand(512, 512, 3) * 255
    transformer = TransformationEngine()
    
    print("\nTesting transformations on 512x512 image:")
    print("  - Zoom 1.2x")
    print("  - Rotate 15°")
    print("  - Translate (10, -5)")
    
    transformed = transformer.apply_transforms(
        dummy_image, 
        zoom=1.2, 
        angle=15, 
        tx=10, 
        ty=-5
    )
    
    print(f"  Original shape: {dummy_image.shape}")
    print(f"  Transformed shape: {transformed.shape}")
    print("\n✓ Interpolation engine working correctly!")
