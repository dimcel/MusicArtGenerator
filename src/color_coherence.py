"""
Simple color coherence utilities.

Current method:
- match_frame0_lab: match LAB channel statistics to a reference frame.
"""

from typing import Literal

import cv2
import numpy as np
from PIL import Image


def _lab_mean_std(arr_rgb: np.ndarray):
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(axis=0)
    std = lab.reshape(-1, 3).std(axis=0)
    std = np.maximum(std, 1e-6)
    return lab, mean, std


def match_frame0_lab(reference: Image.Image, current: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Match current frame color statistics to reference frame in LAB space.

    strength:
      0.0 -> no change
      1.0 -> full LAB-stat matching
    """
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return current

    ref_rgb = np.array(reference.convert("RGB"))
    cur_rgb = np.array(current.convert("RGB"))

    cur_lab, cur_mean, cur_std = _lab_mean_std(cur_rgb)
    _, ref_mean, ref_std = _lab_mean_std(ref_rgb)

    matched = (cur_lab - cur_mean) * (ref_std / cur_std) + ref_mean
    matched = np.clip(matched, 0.0, 255.0).astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    matched_img = Image.fromarray(matched_rgb)

    if s >= 1.0:
        return matched_img
    return Image.blend(current, matched_img, s)


def apply_color_coherence(
    method: Literal["none", "match_frame0_lab"],
    reference_frame: Image.Image,
    current_frame: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    if method == "none":
        return current_frame
    if method == "match_frame0_lab":
        return match_frame0_lab(reference_frame, current_frame, strength=strength)
    return current_frame
