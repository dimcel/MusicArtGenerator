"""
Simple image noise utilities.
"""

from typing import Optional

import numpy as np
from PIL import Image


def add_gaussian_noise(
    image: Image.Image,
    amount: float,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Add gaussian noise to image.

    amount:
      0.0 -> no noise
      1.0 -> very strong noise
    """
    a = float(max(0.0, min(1.0, amount)))
    if a <= 0.0:
        return image

    arr = np.asarray(image).astype(np.float32)
    rng = np.random.default_rng(seed)
    sigma = 35.0 * a
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out)

