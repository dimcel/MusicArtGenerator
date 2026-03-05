"""
ControlNet conditioning image builders.
"""

from PIL import Image
import cv2
import numpy as np


def build_canny_control_image(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """
    Build a 3-channel canny edge control image from an RGB frame.
    """
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low_threshold), int(high_threshold))
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)
