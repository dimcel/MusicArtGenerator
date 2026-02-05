"""
Frame Interpolator - Generates in-between frames to reduce diffusion cost

Supported methods:
- blend: Simple alpha blending (ghosting effect)
- optical_flow: OpenCV optical flow warping (motion-aware)
- rife: Neural interpolation (future, requires installation)
- film: Google's neural interpolation (future, requires installation)
"""

import numpy as np
import cv2
from PIL import Image
from typing import List


class FrameInterpolator:
    """
    Interpolates frames between keyframes to reduce generation cost.
    
    Current implementation: Simple alpha blending
    
    Usage:
        interpolator = FrameInterpolator()
        
        # Generate 3 frames between frame_a and frame_b
        in_between = interpolator.interpolate(frame_a, frame_b, num_frames=3)
        # Result: [frame at 25%, frame at 50%, frame at 75%]
    """
    
    def __init__(self, method: str = "blend"):
        """
        Initialize interpolator.
        
        Args:
            method: Interpolation method
                - "blend": Simple alpha blending (ghosting effect)
                - "optical_flow": OpenCV optical flow (motion-aware)
                - "rife": RIFE neural interpolation (future)
                - "film": FILM neural interpolation (future)
        """
        self.method = method
        
        # Validate method
        valid_methods = ["blend", "optical_flow", "rife", "film"]
        if method not in valid_methods:
            print(f"⚠️  Unknown method '{method}', using 'blend'")
            self.method = "blend"
    
    def interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int
    ) -> List[Image.Image]:
        """
        Generate in-between frames.
        
        Args:
            frame_a: First keyframe
            frame_b: Second keyframe
            num_frames: Number of frames to generate between a and b
            
        Returns:
            List of interpolated frames (does not include frame_a or frame_b)
            
        Example:
            interpolate(frame_0, frame_4, num_frames=3)
            Returns: [frame_1, frame_2, frame_3]
        """
        if self.method == "blend":
            return self._blend_interpolate(frame_a, frame_b, num_frames)
        elif self.method == "optical_flow":
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)
        elif self.method == "rife":
            return self._rife_interpolate(frame_a, frame_b, num_frames)
        elif self.method == "film":
            return self._film_interpolate(frame_a, frame_b, num_frames)
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented")
    
    def _blend_interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int
    ) -> List[Image.Image]:
        """
        Simple alpha blending interpolation.
        
        Formula: frame_i = frame_a * (1 - alpha) + frame_b * alpha
        where alpha goes from 0 to 1
        """
        # Convert to numpy arrays
        arr_a = np.array(frame_a, dtype=np.float32)
        arr_b = np.array(frame_b, dtype=np.float32)
        
        interpolated_frames = []
        
        # Generate frames at evenly spaced positions
        for i in range(1, num_frames + 1):
            # Alpha goes from 1/(num_frames+1) to num_frames/(num_frames+1)
            alpha = i / (num_frames + 1)
            
            # Blend
            blended = arr_a * (1 - alpha) + arr_b * alpha
            
            # Convert back to PIL Image
            blended_img = Image.fromarray(blended.astype(np.uint8))
            interpolated_frames.append(blended_img)
        
        return interpolated_frames


    def _optical_flow_interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int
    ) -> List[Image.Image]:
        """
        Optical flow-based interpolation using OpenCV.
        
        Estimates motion between frames and warps pixels accordingly.
        Better than blending for scenes with motion.
        """
        # Convert to numpy arrays
        arr_a = np.array(frame_a)
        arr_b = np.array(frame_b)
        
        # Convert to grayscale for flow calculation
        gray_a = cv2.cvtColor(arr_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(arr_b, cv2.COLOR_RGB2GRAY)
        
        # Calculate optical flow (motion vectors from a to b)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b,
            None,
            pyr_scale=0.5,    # Image pyramid scale
            levels=3,         # Number of pyramid layers
            winsize=15,       # Averaging window size
            iterations=3,     # Iterations at each pyramid level
            poly_n=5,         # Polynomial expansion neighborhood
            poly_sigma=1.2,   # Gaussian standard deviation
            flags=0
        )
        
        interpolated_frames = []
        
        # Generate frames at evenly spaced positions
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            
            # Create flow map for this interpolation position
            # Scale flow by alpha (partial motion)
            flow_scaled = flow * alpha
            
            # Create coordinate grid
            h, w = arr_a.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Apply flow to coordinates
            x_new = (x + flow_scaled[..., 0]).astype(np.float32)
            y_new = (y + flow_scaled[..., 1]).astype(np.float32)
            
            # Warp frame_a using the flow
            warped_a = cv2.remap(
                arr_a, x_new, y_new,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Blend warped frame with frame_b for smooth transition
            # This handles areas where flow estimation fails
            blended = (warped_a * (1 - alpha) + arr_b * alpha).astype(np.uint8)
            
            # Convert back to PIL Image
            interpolated_img = Image.fromarray(blended)
            interpolated_frames.append(interpolated_img)
        
        return interpolated_frames


    def _rife_interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int
    ) -> List[Image.Image]:
        """
        RIFE neural interpolation (requires torch and RIFE model).
        Falls back to optical_flow if not available.
        """
        try:
            import torch
            from torch.nn import functional as F
            
            # Note: This is a simplified implementation
            # Full RIFE requires: git clone https://github.com/hzwer/arXiv2020-RIFE
            print("⚠️  RIFE model not fully implemented - using optical_flow fallback")
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)
            
            # TODO: Implement full RIFE when model files are available
            # from RIFE.model.RIFE_HD import Model
            # if not hasattr(self, '_rife_model'):
            #     self._rife_model = Model()
            #     self._rife_model.load_model('train_log', -1)
            #     self._rife_model.eval()
            #     self._rife_model.device()
            
        except ImportError as e:
            print(f"⚠️  RIFE dependencies not installed ({e}) - using optical_flow fallback")
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)


    def _film_interpolate(
        self,
        frame_a: Image.Image,
        frame_b: Image.Image,
        num_frames: int
    ) -> List[Image.Image]:
        """
        FILM neural interpolation (requires tensorflow).
        Falls back to optical_flow if not available.
        
        Note: FILM model is large (~73MB) and will download on first use.
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Load FILM model (lazy loading)
            if not hasattr(self, '_film_model'):
                print("📦 Loading FILM model from TensorFlow Hub (first time only)...")
                print("   This may take a minute to download (~73MB)...")
                self._film_model = hub.load("https://tfhub.dev/google/film/1")
                print("   ✓ Model loaded and cached")
            
            # Convert PIL to numpy arrays (RGB, float32, 0-1 range)
            arr_a = np.array(frame_a, dtype=np.float32) / 255.0
            arr_b = np.array(frame_b, dtype=np.float32) / 255.0
            
            # Add batch dimension [1, H, W, 3]
            img0 = tf.expand_dims(tf.convert_to_tensor(arr_a), 0)
            img1 = tf.expand_dims(tf.convert_to_tensor(arr_b), 0)
            
            interpolated_frames = []
            
            # Generate frames at evenly spaced timesteps
            for i in range(1, num_frames + 1):
                alpha = i / (num_frames + 1)
                
                # FILM expects inputs as a dictionary
                # time must be shape (batch, 1) not (batch,)
                inputs = {
                    'x0': img0,
                    'x1': img1,
                    'time': tf.constant([[alpha]], dtype=tf.float32)  # Shape: (1, 1)
                }
                
                # Run FILM interpolation
                output = self._film_model(inputs)
                
                # Extract interpolated image
                # FILM returns dict with 'image' key
                if isinstance(output, dict):
                    interpolated = output['image']
                else:
                    interpolated = output
                
                # Convert back to PIL (remove batch dim, scale to 0-255, convert to uint8)
                output_np = (interpolated.numpy()[0] * 255.0).clip(0, 255).astype(np.uint8)
                interpolated_img = Image.fromarray(output_np)
                interpolated_frames.append(interpolated_img)
                
                if (i % 10 == 0 or i == num_frames):
                    print(f"      FILM interpolated {i}/{num_frames} frames")
            
            return interpolated_frames
            
        except ImportError as e:
            print(f"⚠️  FILM dependencies not installed ({e})")
            print("   Install with: pip install tensorflow tensorflow-hub")
            print("   Falling back to optical_flow")
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)
        except Exception as e:
            print(f"⚠️  FILM error: {e}")
            print("   Falling back to optical_flow")
            import traceback
            traceback.print_exc()
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)


def simple_interpolate(
    frame_a: Image.Image,
    frame_b: Image.Image,
    num_frames: int = 1
) -> List[Image.Image]:
    """
    Quick helper function for simple interpolation.
    
    Args:
        frame_a: First frame
        frame_b: Second frame
        num_frames: How many frames to generate between them
        
    Returns:
        List of interpolated frames
        
    Example:
        # Generate 1 frame between frame_0 and frame_2
        frame_1 = simple_interpolate(frame_0, frame_2, num_frames=1)[0]
    """
    interpolator = FrameInterpolator(method="blend")
    return interpolator.interpolate(frame_a, frame_b, num_frames)


# =============================================================================
# FUTURE ENHANCEMENTS - NOT YET IMPLEMENTED
# =============================================================================
#
# Option 1: RIFE (Real-Time Intermediate Flow Estimation)
# --------------------------------------------------------
# - Fast neural interpolation (real-time on GPU)
# - Better quality than optical flow, handles occlusions
# - Installation: pip install torch torchvision
#                 git clone https://github.com/hzwer/arXiv2020-RIFE
# - Implementation:
#     def _rife_interpolate(self, frame_a, frame_b, num_frames):
#         from RIFE.model.RIFE_HD import Model
#         model = Model()
#         model.load_model('train_log', -1)
#         model.eval()
#         
#         # Convert to tensors
#         img0 = torch.from_numpy(np.array(frame_a)).permute(2,0,1).float() / 255
#         img1 = torch.from_numpy(np.array(frame_b)).permute(2,0,1).float() / 255
#         
#         frames = []
#         for i in range(1, num_frames + 1):
#             timestep = i / (num_frames + 1)
#             output = model.inference(img0, img1, timestep)
#             frames.append(Image.fromarray(output))
#         return frames
#
# Option 2: FILM (Frame Interpolation for Large Motion)
# -------------------------------------------------------
# - Google's state-of-the-art interpolation
# - Best quality, handles complex motion and large displacements
# - Installation: pip install tensorflow tensorflow-hub
# - Implementation:
#     def _film_interpolate(self, frame_a, frame_b, num_frames):
#         import tensorflow as tf
#         import tensorflow_hub as hub
#         
#         model = hub.load("https://tfhub.dev/google/film/1")
#         
#         # Prepare inputs
#         img0 = tf.convert_to_tensor(np.array(frame_a)) / 255.0
#         img1 = tf.convert_to_tensor(np.array(frame_b)) / 255.0
#         
#         frames = []
#         for i in range(1, num_frames + 1):
#             timestep = i / (num_frames + 1)
#             output = model(img0, img1, timestep)
#             frames.append(Image.fromarray((output.numpy() * 255).astype(np.uint8)))
#         return frames
#
# Implementation Strategy:
# ------------------------
# 1. Add lazy loading for models (only load when method is used)
# 2. Add try/except for optional dependencies
# 3. Fall back to optical_flow or blend if packages not installed
# 4. Cache loaded models to avoid reloading on each call
#
# Example with fallback:
#     def _rife_interpolate(self, ...):
#         try:
#             from RIFE.model.RIFE_HD import Model
#             # ... RIFE code ...
#         except ImportError:
#             print("⚠️  RIFE not installed, falling back to optical_flow")
#             return self._optical_flow_interpolate(...)
#
# =============================================================================
