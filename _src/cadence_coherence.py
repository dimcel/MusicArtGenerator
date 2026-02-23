"""
Cadence coherence helper for skipped diffusion frames.

When cadence mode skips img2img on a frame, use this helper to reduce
visual jitter/drift by blending or optical-flow warping.
"""

from typing import Literal

import numpy as np
from PIL import Image


class CadenceCoherence:
    def __init__(
        self,
        method: Literal["none", "blend", "optical_flow"] = "blend",
        blend_alpha: float = 0.35,
    ):
        self.method = method
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))

    def apply(self, prev_frame: Image.Image, transformed_frame: Image.Image) -> Image.Image:
        if self.method == "none":
            return transformed_frame
        if self.method == "blend":
            return Image.blend(prev_frame, transformed_frame, self.blend_alpha)
        if self.method == "optical_flow":
            return self._optical_flow_mix(prev_frame, transformed_frame)
        return transformed_frame

    def _optical_flow_mix(self, prev_frame: Image.Image, transformed_frame: Image.Image) -> Image.Image:
        try:
            import cv2
        except Exception:
            return Image.blend(prev_frame, transformed_frame, self.blend_alpha)

        arr_prev = np.array(prev_frame)
        arr_tr = np.array(transformed_frame)

        try:
            gray_prev = cv2.cvtColor(arr_prev, cv2.COLOR_RGB2GRAY)
            gray_tr = cv2.cvtColor(arr_tr, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_tr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            h, w = gray_prev.shape
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (xx + flow[..., 0]).astype(np.float32)
            map_y = (yy + flow[..., 1]).astype(np.float32)
            warped_prev = cv2.remap(
                arr_prev,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            a = self.blend_alpha
            mixed = (warped_prev * (1.0 - a) + arr_tr * a).astype(np.uint8)
            return Image.fromarray(mixed)
        except Exception:
            return Image.blend(prev_frame, transformed_frame, self.blend_alpha)

