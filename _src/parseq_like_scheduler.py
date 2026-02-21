"""
Parseq-like Scheduler (Simple)

Minimal per-frame scheduling engine inspired by sd-parseq:
- Keyframe curves (interpolated values over time)
- Optional expressions (computed each frame)
- Optional beat pulse modulation

This is intentionally small and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import math
from typing import Dict, Iterable, List, Optional, Tuple


EASING_MODES = {"linear", "ease_in", "ease_out", "ease_in_out", "step"}


@dataclass
class ParameterSpec:
    """Defines how one parameter should evolve over frames."""

    keyframes: Optional[List[Tuple[int, float]]] = None
    expression: Optional[str] = None
    default: float = 0.0
    easing: str = "linear"
    beat_pulse_amount: float = 0.0
    beat_pulse_decay_frames: int = 0
    clamp: Optional[Tuple[float, float]] = None


def _ease(t: float, mode: str) -> float:
    t = max(0.0, min(1.0, t))
    if mode == "linear":
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode == "ease_in_out":
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
    if mode == "step":
        return 0.0
    return t


def _interpolate_keyframes(
    frame: int,
    keyframes: List[Tuple[int, float]],
    easing: str,
) -> float:
    if not keyframes:
        raise ValueError("keyframes cannot be empty")

    keyframes = sorted(keyframes, key=lambda x: x[0])

    if frame <= keyframes[0][0]:
        return keyframes[0][1]
    if frame >= keyframes[-1][0]:
        return keyframes[-1][1]

    left = keyframes[0]
    right = keyframes[-1]
    for i in range(len(keyframes) - 1):
        a = keyframes[i]
        b = keyframes[i + 1]
        if a[0] <= frame <= b[0]:
            left, right = a, b
            break

    span = max(1, right[0] - left[0])
    t = (frame - left[0]) / span
    tt = _ease(t, easing)
    return left[1] + (right[1] - left[1]) * tt


def _nearest_beat_pulse(frame: int, beat_frames: List[int], decay_frames: int) -> float:
    if decay_frames <= 0 or not beat_frames:
        return 0.0

    best = 0.0
    for b in beat_frames:
        d = abs(frame - b)
        if d <= decay_frames:
            # Triangular pulse: 1 on beat, then linearly decays to 0.
            pulse = 1.0 - (d / decay_frames)
            if pulse > best:
                best = pulse
    return best


def _safe_eval(expr: str, variables: Dict[str, float]) -> float:
    allowed_names = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
    }
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in variables:
                raise ValueError(f"Unsupported variable: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                raise ValueError("Only whitelisted functions are allowed in expressions")

    compiled = compile(tree, "<schedule_expr>", "eval")
    env = dict(allowed_names)
    env.update(variables)
    return float(eval(compiled, {"__builtins__": {}}, env))


class ParseqLikeScheduler:
    """
    Tiny per-frame scheduler:
    - Parameter curves from keyframes
    - Optional per-frame expression
    - Optional beat pulse contribution
    """

    def __init__(self, specs: Dict[str, ParameterSpec], beat_frames: Optional[Iterable[int]] = None):
        self.specs = specs
        self.beat_frames = sorted(set(int(f) for f in (beat_frames or [])))
        for name, spec in self.specs.items():
            if spec.easing not in EASING_MODES:
                raise ValueError(f"{name}: easing must be one of {sorted(EASING_MODES)}")

    def value_for(self, param_name: str, frame: int, total_frames: int) -> float:
        if param_name not in self.specs:
            raise KeyError(f"Unknown parameter: {param_name}")
        spec = self.specs[param_name]

        t = 0.0 if total_frames <= 1 else frame / (total_frames - 1)
        beat = 1.0 if frame in self.beat_frames else 0.0
        beat_pulse = _nearest_beat_pulse(frame, self.beat_frames, spec.beat_pulse_decay_frames)

        if spec.expression:
            value = _safe_eval(
                spec.expression,
                {
                    "t": t,
                    "frame": float(frame),
                    "total_frames": float(total_frames),
                    "beat": beat,
                    "beat_pulse": beat_pulse,
                },
            )
        elif spec.keyframes:
            value = _interpolate_keyframes(frame, spec.keyframes, spec.easing)
        else:
            value = spec.default

        if spec.beat_pulse_amount != 0.0:
            value += spec.beat_pulse_amount * beat_pulse

        if spec.clamp:
            lo, hi = spec.clamp
            value = max(lo, min(hi, value))

        return float(value)

    def frame_params(self, frame: int, total_frames: int) -> Dict[str, float]:
        return {name: self.value_for(name, frame, total_frames) for name in self.specs}

    def build_schedule(self, total_frames: int) -> List[Dict[str, float]]:
        return [self.frame_params(f, total_frames) for f in range(total_frames)]

