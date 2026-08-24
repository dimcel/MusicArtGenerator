"""
Cadence coherence helper for skipped diffusion frames.

When cadence mode skips img2img on a frame, use this helper to reduce
visual jitter/drift by blending or by delegating to FrameInterpolator
methods (optical_flow/rife/film).
"""

from PIL import Image
from typing import Literal, Optional

from .frame_interpolator import FrameInterpolator


class CadenceCoherence:
    def __init__(
        self,
        method: Literal["none", "blend", "optical_flow", "rife", "film"] = "blend",
        blend_alpha: float = 0.35,
    ):
        self.method = str(method).strip().lower()
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))
        self._interpolator: Optional[FrameInterpolator] = None

        valid_methods = {"none", "blend", "optical_flow", "rife", "film"}
        if self.method not in valid_methods:
            print(f"⚠️  Unknown cadence coherence method '{self.method}', using 'blend'")
            self.method = "blend"

        # Keep 'none' and 'blend' lightweight; interpolation engines are used
        # for optical_flow/rife/film paths.
        if self.method in {"optical_flow", "rife", "film"}:
            self._interpolator = FrameInterpolator(method=self.method)

    def apply(self, prev_frame: Image.Image, transformed_frame: Image.Image) -> Image.Image:
        if self.method == "none":
            return transformed_frame
        if self.method == "blend":
            return Image.blend(prev_frame, transformed_frame, self.blend_alpha)

        if self._interpolator is None:
            return Image.blend(prev_frame, transformed_frame, self.blend_alpha)

        try:
            # Cadence skip fills exactly one frame each step.
            frames = self._interpolator.interpolate(prev_frame, transformed_frame, num_frames=1)
            if frames:
                return frames[0]
        except Exception:
            pass
        return Image.blend(prev_frame, transformed_frame, self.blend_alpha)
