"""
Image Transformation Module
Applies 2D transformations (zoom, pan, rotate) to images before diffusion

Similar to Deforum's approach: transform the previous frame BEFORE 
feeding it into img2img diffusion.

Usage:
    from music_art_generator.image_transform import transform_image
    
    # Zoom in slightly
    transformed = transform_image(image, zoom=1.05)
    
    # Pan right
    transformed = transform_image(image, translation_x=10)
    
    # Combine effects
    transformed = transform_image(image, zoom=1.1, angle=2.0, translation_x=5)
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple


def transform_image(
    image: Image.Image,
    zoom: float = 1.0,
    angle: float = 0.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0
) -> Image.Image:
    """
    Apply 2D transformations to an image.
    
    This is applied BEFORE diffusion (like Deforum does):
    1. Take previous frame
    2. Apply transformations (zoom/pan/rotate)
    3. Feed transformed frame into img2img
    
    Args:
        image: Input PIL Image
        zoom: Zoom factor
            - 1.0 = no change
            - >1.0 = zoom in (e.g., 1.05 = zoom in 5%)
            - <1.0 = zoom out (e.g., 0.95 = zoom out 5%)
        angle: Rotation angle in degrees (positive = clockwise)
        translation_x: Horizontal shift in pixels (positive = right)
        translation_y: Vertical shift in pixels (positive = down)
        
    Returns:
        Transformed PIL Image
        
    Example:
        # Zoom in 10% and pan right 15 pixels
        new_frame = transform_image(prev_frame, zoom=1.1, translation_x=15)
    """
    # Convert to numpy
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Center point
    center_x = w / 2
    center_y = h / 2
    
    # Create transformation matrix
    # OpenCV uses (x, y) format, so center is (center_x, center_y)
    
    # Step 1: Rotation + Zoom combined
    # getRotationMatrix2D creates rotation around center with scale
    scale = zoom  # OpenCV scale is same as zoom
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    
    # Step 2: Add translation
    M[0, 2] += translation_x  # x translation
    M[1, 2] += translation_y  # y translation
    
    # Apply transformation
    transformed = cv2.warpAffine(
        img_array,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # Replicate edge pixels to avoid black borders
    )
    
    # Convert back to PIL
    return Image.fromarray(transformed)


def transform_parameters_from_beat(
    is_beat: bool,
    frame_num: int,
    zoom_per_frame: float = 0.01,
    beat_zoom_boost: float = 0.03
) -> dict:
    """
    Generate transform parameters that can be synced with beats.
    
    Simple example of how to make zoom react to music beats.
    
    Args:
        is_beat: Whether current frame is on a beat
        frame_num: Current frame number
        zoom_per_frame: Base zoom increment per frame (default: 1% per frame)
        beat_zoom_boost: Extra zoom on beats (default: 3% boost)
        
    Returns:
        Dictionary with transform parameters
        
    Example:
        params = transform_parameters_from_beat(
            is_beat=True,
            frame_num=24
        )
        transformed = transform_image(image, **params)
    """
    # Base zoom increases over time
    base_zoom = 1.0 + (frame_num * zoom_per_frame)
    
    # Add extra zoom on beats
    if is_beat:
        zoom = base_zoom + beat_zoom_boost
    else:
        zoom = base_zoom
    
    return {
        'zoom': zoom,
        'angle': 0.0,
        'translation_x': 0.0,
        'translation_y': 0.0
    }


# =============================================================================
# SIMPLE HELPER FUNCTIONS
# =============================================================================

def zoom_in(image: Image.Image, amount: float = 1.05) -> Image.Image:
    """Quick helper: zoom in by amount"""
    return transform_image(image, zoom=amount)


def zoom_out(image: Image.Image, amount: float = 0.95) -> Image.Image:
    """Quick helper: zoom out by amount"""
    return transform_image(image, zoom=amount)


def pan_right(image: Image.Image, pixels: float = 10) -> Image.Image:
    """Quick helper: pan right"""
    return transform_image(image, translation_x=pixels)


def pan_left(image: Image.Image, pixels: float = 10) -> Image.Image:
    """Quick helper: pan left"""
    return transform_image(image, translation_x=-pixels)


def pan_down(image: Image.Image, pixels: float = 10) -> Image.Image:
    """Quick helper: pan down"""
    return transform_image(image, translation_y=pixels)


def pan_up(image: Image.Image, pixels: float = 10) -> Image.Image:
    """Quick helper: pan up"""
    return transform_image(image, translation_y=-pixels)


def rotate_clockwise(image: Image.Image, degrees: float = 5) -> Image.Image:
    """Quick helper: rotate clockwise"""
    return transform_image(image, angle=degrees)


def rotate_counterclockwise(image: Image.Image, degrees: float = 5) -> Image.Image:
    """Quick helper: rotate counterclockwise"""
    return transform_image(image, angle=-degrees)
