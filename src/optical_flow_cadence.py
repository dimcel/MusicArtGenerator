"""
Optical Flow Cadence - Deforum's Clever Frame Interpolation Technique

This module implements the key technique that makes Deforum animations smooth:
- Generate fewer diffused frames (saves time)  
- Use optical flow to create intelligent in-between frames
- Apply color coherence to prevent drift

Instead of diffusing every frame:
    Frame 0 [DIFFUSE] → Frame 1 [DIFFUSE] → Frame 2 [DIFFUSE] → ...  (slow!)

With cadence=4:
    Frame 0 [DIFFUSE] → warp → warp → warp → Frame 4 [DIFFUSE] → ...  (4x faster!)

The magic is in HOW we warp - using optical flow to track motion.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
from enum import Enum


class OpticalFlowMethod(Enum):
    """Available optical flow algorithms"""
    FARNEBACK = "farneback"      # Fast, good for simple motion
    DIS_MEDIUM = "dis_medium"    # Balanced speed/quality  
    DIS_FINE = "dis_fine"        # Higher quality, slower
    RAFT = "raft"                # Best quality, requires torch (not implemented here)


class ColorCoherenceMethod(Enum):
    """Color matching methods to prevent drift"""
    NONE = "none"
    LAB = "lab"           # Best for most cases
    HSV = "hsv"           # Good for saturated images
    RGB = "rgb"           # Simple but effective
    MATCH_FRAME_0 = "match_frame_0"  # Always match to first frame


@dataclass
class CadenceConfig:
    """Configuration for cadence-based rendering"""
    
    # Core cadence settings
    cadence: int = 4                    # Diffuse every Nth frame (1=every frame, 4=every 4th)
    
    # Optical flow settings
    use_optical_flow: bool = True       # Use optical flow for in-betweens (vs simple blend)
    flow_method: OpticalFlowMethod = OpticalFlowMethod.DIS_MEDIUM
    flow_blend_factor: float = 1.0      # How much to apply flow (0=none, 1=full)
    
    # Color coherence
    color_coherence: ColorCoherenceMethod = ColorCoherenceMethod.LAB
    color_match_frame: int = 0          # Which frame to match colors to
    
    # Blending
    blend_mode: Literal["flow", "linear", "hybrid"] = "flow"
    hybrid_flow_weight: float = 0.7     # For hybrid: weight of flow vs linear blend


class OpticalFlowCalculator:
    """
    Calculates optical flow between frames using various methods.
    
    Optical flow tells us WHERE each pixel moved between two frames.
    Output is a 2D vector field (same size as image) where each pixel
    has (dx, dy) showing its movement.
    """
    
    def __init__(self, method: OpticalFlowMethod = OpticalFlowMethod.DIS_MEDIUM):
        self.method = method
        self._init_algorithm()
    
    def _init_algorithm(self):
        """Initialize the optical flow algorithm"""
        if self.method == OpticalFlowMethod.FARNEBACK:
            # Farneback is built into cv2, no special init needed
            self._algo = None
        elif self.method == OpticalFlowMethod.DIS_MEDIUM:
            self._algo = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.method == OpticalFlowMethod.DIS_FINE:
            self._algo = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            # Override to FINE settings
            self._algo.setFinestScale(1)
            self._algo.setPatchSize(8)
            self._algo.setPatchStride(4)
        else:
            self._algo = None
    
    def calculate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow from frame1 to frame2.
        
        Args:
            frame1: First frame (H, W, 3) uint8
            frame2: Second frame (H, W, 3) uint8
            
        Returns:
            Flow field (H, W, 2) float32 with (dx, dy) per pixel
        """
        # Convert to grayscale for flow calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        if self.method == OpticalFlowMethod.FARNEBACK:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, 
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            # DIS methods
            flow = self._algo.calc(gray1, gray2, None)
        
        return flow
    
    def warp_frame(self, frame: np.ndarray, flow: np.ndarray, 
                   flow_factor: float = 1.0) -> np.ndarray:
        """
        Warp a frame using optical flow.
        
        Args:
            frame: Image to warp (H, W, 3)
            flow: Flow field (H, W, 2) 
            flow_factor: How much to apply (0=no warp, 1=full warp, 0.5=half)
            
        Returns:
            Warped frame (H, W, 3)
        """
        h, w = frame.shape[:2]
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow (scaled by factor)
        map_x = (x + flow[:, :, 0] * flow_factor).astype(np.float32)
        map_y = (y + flow[:, :, 1] * flow_factor).astype(np.float32)
        
        # Remap (warp) the image
        warped = cv2.remap(frame, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        
        return warped


class ColorCoherence:
    """
    Maintains color consistency across frames.
    
    Without this, colors can drift significantly over many frames,
    leading to washed out or oversaturated results.
    """
    
    def __init__(self, method: ColorCoherenceMethod = ColorCoherenceMethod.LAB):
        self.method = method
        self.reference_frame: Optional[np.ndarray] = None
        self.reference_stats: Optional[dict] = None
    
    def set_reference(self, frame: np.ndarray):
        """Set the reference frame for color matching"""
        self.reference_frame = frame.copy()
        self.reference_stats = self._compute_stats(frame)
    
    def _compute_stats(self, frame: np.ndarray) -> dict:
        """Compute color statistics for a frame"""
        if self.method == ColorCoherenceMethod.LAB:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
            return {
                'mean': np.mean(lab, axis=(0, 1)),
                'std': np.std(lab, axis=(0, 1))
            }
        elif self.method == ColorCoherenceMethod.HSV:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            return {
                'mean': np.mean(hsv, axis=(0, 1)),
                'std': np.std(hsv, axis=(0, 1))
            }
        else:  # RGB
            return {
                'mean': np.mean(frame.astype(np.float32), axis=(0, 1)),
                'std': np.std(frame.astype(np.float32), axis=(0, 1))
            }
    
    def match(self, frame: np.ndarray) -> np.ndarray:
        """
        Match frame's colors to the reference frame.
        
        Uses histogram matching in the selected color space.
        """
        if self.method == ColorCoherenceMethod.NONE or self.reference_stats is None:
            return frame
        
        # Get current frame stats
        current_stats = self._compute_stats(frame)
        
        if self.method == ColorCoherenceMethod.LAB:
            return self._match_lab(frame, current_stats)
        elif self.method == ColorCoherenceMethod.HSV:
            return self._match_hsv(frame, current_stats)
        else:
            return self._match_rgb(frame, current_stats)
    
    def _match_lab(self, frame: np.ndarray, current_stats: dict) -> np.ndarray:
        """Match colors in LAB space (best for most cases)"""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Normalize and denormalize to match reference
        ref_mean, ref_std = self.reference_stats['mean'], self.reference_stats['std']
        cur_mean, cur_std = current_stats['mean'], current_stats['std']
        
        # Avoid division by zero
        cur_std = np.where(cur_std < 1e-6, 1, cur_std)
        
        # Match: (x - cur_mean) / cur_std * ref_std + ref_mean
        lab = (lab - cur_mean) / cur_std * ref_std + ref_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _match_hsv(self, frame: np.ndarray, current_stats: dict) -> np.ndarray:
        """Match colors in HSV space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        ref_mean, ref_std = self.reference_stats['mean'], self.reference_stats['std']
        cur_mean, cur_std = current_stats['mean'], current_stats['std']
        cur_std = np.where(cur_std < 1e-6, 1, cur_std)
        
        # Only match S and V, keep H (hue) mostly unchanged
        hsv[:, :, 1:] = (hsv[:, :, 1:] - cur_mean[1:]) / cur_std[1:] * ref_std[1:] + ref_mean[1:]
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def _match_rgb(self, frame: np.ndarray, current_stats: dict) -> np.ndarray:
        """Match colors in RGB space (simple but effective)"""
        frame_f = frame.astype(np.float32)
        
        ref_mean, ref_std = self.reference_stats['mean'], self.reference_stats['std']
        cur_mean, cur_std = current_stats['mean'], current_stats['std']
        cur_std = np.where(cur_std < 1e-6, 1, cur_std)
        
        matched = (frame_f - cur_mean) / cur_std * ref_std + ref_mean
        return np.clip(matched, 0, 255).astype(np.uint8)


class CadenceInterpolator:
    """
    The main class that implements optical flow cadence.
    
    This is the "clever" part of Deforum - instead of diffusing every frame,
    we diffuse fewer frames and use optical flow to create smooth in-betweens.
    """
    
    def __init__(self, config: CadenceConfig = None):
        self.config = config or CadenceConfig()
        self.flow_calculator = OpticalFlowCalculator(self.config.flow_method)
        self.color_coherence = ColorCoherence(self.config.color_coherence)
        
        # Storage for cadence rendering
        self._prev_diffused_frame: Optional[np.ndarray] = None
        self._next_diffused_frame: Optional[np.ndarray] = None
        self._flow_forward: Optional[np.ndarray] = None
        self._flow_backward: Optional[np.ndarray] = None
    
    def should_diffuse(self, frame_idx: int) -> bool:
        """
        Returns True if this frame should be diffused (run through SD).
        
        With cadence=4:
            Frame 0: True  (diffuse)
            Frame 1: False (interpolate)
            Frame 2: False (interpolate)
            Frame 3: False (interpolate)
            Frame 4: True  (diffuse)
            ...
        """
        return frame_idx % self.config.cadence == 0
    
    def get_diffusion_frames(self, total_frames: int) -> List[int]:
        """Get list of frame indices that need diffusion"""
        return [i for i in range(total_frames) if self.should_diffuse(i)]
    
    def set_diffused_frames(self, prev_frame: np.ndarray, next_frame: np.ndarray):
        """
        Set the two diffused keyframes for interpolation.
        
        Args:
            prev_frame: The earlier diffused frame (RGB, uint8)
            next_frame: The later diffused frame (RGB, uint8)
        """
        self._prev_diffused_frame = np.array(prev_frame)
        self._next_diffused_frame = np.array(next_frame)
        
        # Compute bidirectional optical flow
        if self.config.use_optical_flow:
            print("    Computing optical flow...")
            self._flow_forward = self.flow_calculator.calculate(
                self._prev_diffused_frame, 
                self._next_diffused_frame
            )
            self._flow_backward = self.flow_calculator.calculate(
                self._next_diffused_frame,
                self._prev_diffused_frame
            )
        
        # Set color reference from first frame (if not already set)
        if self.color_coherence.reference_frame is None:
            self.color_coherence.set_reference(prev_frame)
    
    def interpolate_frame(self, position: float) -> np.ndarray:
        """
        Generate an intermediate frame at the given position.
        
        Args:
            position: 0.0 = prev_frame, 1.0 = next_frame, 0.5 = middle
            
        Returns:
            Interpolated frame (RGB, uint8)
        """
        if self._prev_diffused_frame is None or self._next_diffused_frame is None:
            raise ValueError("Must call set_diffused_frames() first")
        
        if self.config.blend_mode == "linear" or not self.config.use_optical_flow:
            # Simple linear blend
            result = self._linear_blend(position)
        elif self.config.blend_mode == "flow":
            # Pure optical flow warping
            result = self._flow_warp(position)
        else:  # hybrid
            # Mix of flow and linear
            flow_result = self._flow_warp(position)
            linear_result = self._linear_blend(position)
            w = self.config.hybrid_flow_weight
            result = (flow_result * w + linear_result * (1 - w)).astype(np.uint8)
        
        # Apply color coherence
        if self.config.color_coherence != ColorCoherenceMethod.NONE:
            result = self.color_coherence.match(result)
        
        return result
    
    def _linear_blend(self, position: float) -> np.ndarray:
        """Simple linear interpolation between frames"""
        return cv2.addWeighted(
            self._prev_diffused_frame, 1 - position,
            self._next_diffused_frame, position,
            0
        )
    
    def _flow_warp(self, position: float) -> np.ndarray:
        """
        Warp both frames toward the target position using optical flow.
        
        The key insight: we warp BOTH frames toward the middle and blend them.
        - Warp prev_frame forward by (position * flow)
        - Warp next_frame backward by ((1-position) * flow)
        - Blend the two warped frames
        """
        # Warp prev_frame forward
        warped_prev = self.flow_calculator.warp_frame(
            self._prev_diffused_frame,
            self._flow_forward,
            flow_factor=position * self.config.flow_blend_factor
        )
        
        # Warp next_frame backward
        warped_next = self.flow_calculator.warp_frame(
            self._next_diffused_frame,
            self._flow_backward,
            flow_factor=(1 - position) * self.config.flow_blend_factor
        )
        
        # Blend the two warped frames
        # Weight shifts: at position=0, mostly prev; at position=1, mostly next
        result = cv2.addWeighted(
            warped_prev, 1 - position,
            warped_next, position,
            0
        )
        
        return result
    
    def generate_cadence_frames(
        self, 
        prev_frame: np.ndarray, 
        next_frame: np.ndarray,
        num_intermediates: int
    ) -> List[np.ndarray]:
        """
        Generate all intermediate frames between two diffused frames.
        
        Args:
            prev_frame: First diffused frame
            next_frame: Second diffused frame  
            num_intermediates: How many frames to generate between them
            
        Returns:
            List of intermediate frames (not including prev_frame and next_frame)
        """
        self.set_diffused_frames(prev_frame, next_frame)
        
        frames = []
        for i in range(1, num_intermediates + 1):
            position = i / (num_intermediates + 1)
            frame = self.interpolate_frame(position)
            frames.append(frame)
        
        return frames


def integrate_cadence_with_render(
    render_func,
    schedule: List[dict],
    config: CadenceConfig,
    **render_kwargs
) -> List[np.ndarray]:
    """
    High-level function to integrate cadence into your existing render loop.
    
    Instead of rendering every frame, renders only cadence frames and
    interpolates the rest.
    
    Args:
        render_func: Your function that renders a single frame
                     signature: render_func(frame_idx, params, prev_frame) -> np.ndarray
        schedule: Your animation schedule (list of per-frame params)
        config: CadenceConfig
        **render_kwargs: Additional kwargs passed to render_func
        
    Returns:
        List of all frames (diffused + interpolated)
    """
    interpolator = CadenceInterpolator(config)
    total_frames = len(schedule)
    diffusion_frames = interpolator.get_diffusion_frames(total_frames)
    
    print(f"\n🎬 Cadence Rendering")
    print(f"   Total frames: {total_frames}")
    print(f"   Diffusion frames: {len(diffusion_frames)} (cadence={config.cadence})")
    print(f"   Speedup: ~{config.cadence}x")
    print(f"   Optical flow: {config.use_optical_flow}")
    print(f"   Color coherence: {config.color_coherence.value}")
    
    all_frames = [None] * total_frames
    prev_diffused = None
    prev_diffused_idx = None
    
    for diff_idx in diffusion_frames:
        params = schedule[diff_idx]
        
        # Render the diffused frame
        print(f"\n[{diff_idx}/{total_frames}] 🎨 DIFFUSING...")
        current_diffused = render_func(diff_idx, params, prev_diffused, **render_kwargs)
        all_frames[diff_idx] = current_diffused
        
        # If we have a previous diffused frame, interpolate between them
        if prev_diffused is not None and prev_diffused_idx is not None:
            num_intermediates = diff_idx - prev_diffused_idx - 1
            if num_intermediates > 0:
                print(f"    Interpolating {num_intermediates} frames...")
                intermediate_frames = interpolator.generate_cadence_frames(
                    prev_diffused, 
                    current_diffused,
                    num_intermediates
                )
                for i, frame in enumerate(intermediate_frames):
                    all_frames[prev_diffused_idx + 1 + i] = frame
        
        prev_diffused = current_diffused
        prev_diffused_idx = diff_idx
    
    # Handle any trailing frames (after last diffusion)
    if prev_diffused_idx is not None and prev_diffused_idx < total_frames - 1:
        # Just duplicate the last frame for remaining
        for i in range(prev_diffused_idx + 1, total_frames):
            all_frames[i] = prev_diffused.copy()
    
    return all_frames


# =============================================================================
# PIL Image helpers
# =============================================================================

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB, uint8)"""
    return np.array(image.convert('RGB'))

def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    return Image.fromarray(array.astype(np.uint8))


# =============================================================================
# Example / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OPTICAL FLOW CADENCE - Demo")
    print("=" * 60)
    
    # Create test frames (simple gradient shift)
    h, w = 256, 256
    
    # Frame 1: Red gradient on left
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w // 2):
        frame1[:, x, 0] = int(255 * x / (w // 2))  # Red gradient
    frame1[:, :, 2] = 100  # Some blue
    
    # Frame 2: Red gradient shifted right
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w // 2, w):
        frame2[:, x, 0] = int(255 * (x - w // 2) / (w // 2))
    frame2[:, :, 2] = 100
    
    print(f"\nTest frames: {frame1.shape}")
    
    # Create interpolator
    config = CadenceConfig(
        cadence=4,
        use_optical_flow=True,
        flow_method=OpticalFlowMethod.DIS_MEDIUM,
        color_coherence=ColorCoherenceMethod.LAB,
        blend_mode="flow"
    )
    
    interpolator = CadenceInterpolator(config)
    
    # Generate intermediate frames
    print("\nGenerating 3 intermediate frames...")
    intermediates = interpolator.generate_cadence_frames(frame1, frame2, num_intermediates=3)
    
    print(f"Generated {len(intermediates)} frames")
    
    # Save test output
    import os
    os.makedirs("/tmp/cadence_test", exist_ok=True)
    
    numpy_to_pil(frame1).save("/tmp/cadence_test/frame_0_diffused.png")
    for i, frame in enumerate(intermediates):
        numpy_to_pil(frame).save(f"/tmp/cadence_test/frame_{i+1}_interpolated.png")
    numpy_to_pil(frame2).save("/tmp/cadence_test/frame_4_diffused.png")
    
    print(f"\n✓ Saved test frames to /tmp/cadence_test/")
    print(f"  frame_0_diffused.png     - First keyframe")
    print(f"  frame_1_interpolated.png - Optical flow interpolated")
    print(f"  frame_2_interpolated.png - Optical flow interpolated")
    print(f"  frame_3_interpolated.png - Optical flow interpolated")
    print(f"  frame_4_diffused.png     - Second keyframe")
    
    # Show stats
    print(f"\n📊 Cadence Statistics:")
    print(f"   With cadence={config.cadence}:")
    print(f"   - 120 frame video needs only {len(interpolator.get_diffusion_frames(120))} diffusions")
    print(f"   - That's {config.cadence}x faster generation!")