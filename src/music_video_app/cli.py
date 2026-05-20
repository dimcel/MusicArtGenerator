"""CLI for the music video app."""

import argparse
import sys
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (mp3/wav/etc.)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for diffusion pipelines (auto, cpu, mps, cuda, cuda:<index>).",
    )
    parser.add_argument("--mode", choices=["full", "cadence"], default="full")
    parser.add_argument("--cadence", type=int, default=3)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help=(
            "User prompt override. Supports single prompt or prompt list with '||' "
            "separator (or JSON list string). In init-image identity mode, this is "
            "the main prompt source."
        ),
    )
    parser.add_argument(
        "--prompt-change-every-beats",
        type=int,
        default=1,
        help="When using prompt lists, switch to next prompt every N detected beats.",
    )
    parser.add_argument(
        "--music-color-fx",
        action="store_true",
        help="Enable simple music-driven color FX (beat pop + warm/cool tint).",
    )
    parser.add_argument(
        "--onset-jitter",
        action="store_true",
        help="Enable small shutter/jitter camera shake from onset spikes.",
    )
    parser.add_argument(
        "--quiet-hold",
        action="store_true",
        help="Detect quiet sections and calm motion (less zoom/pan/angle/noise), pause jitter and prompt switches.",
    )
    parser.add_argument(
        "--color-coherence",
        choices=["none", "match_frame0_lab"],
        default="none",
        help="Optional color coherence post-process.",
    )
    parser.add_argument(
        "--color-coherence-strength",
        type=float,
        default=0.60,
        help="Blend strength for color coherence (0..1).",
    )
    parser.add_argument("--prompt-mode", choices=["blend", "hard"], default="blend")
    parser.add_argument("--coherence", choices=["none", "blend", "optical_flow", "rife", "film"], default="blend")
    parser.add_argument("--subject-hold-frames", type=int, default=32)
    parser.add_argument("--subject-transition-frames", type=int, default=16)
    parser.add_argument("--subject-smoothing-window", type=int, default=41)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=12.0,
        help=(
            "Without --resume-dir: generate first N seconds from start. "
            "With --resume-dir: append N more seconds from last completed frame. "
            "<=0 means full audio."
        ),
    )
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--controlnet", action="store_true", help="Enable ControlNet img2img (canny).")
    parser.add_argument(
        "--controlnet-model",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="ControlNet model id.",
    )
    parser.add_argument("--control-scale-base", type=float, default=0.80)
    parser.add_argument("--control-scale-beat-boost", type=float, default=0.30)
    parser.add_argument("--control-scale-onset-boost", type=float, default=0.20)
    parser.add_argument("--zoom-base", type=float, default=1.002, help="Base per-frame zoom multiplier for test profile.")
    parser.add_argument("--zoom-beat-boost", type=float, default=0.020, help="Additional zoom multiplier from beat pulse for test profile.")
    parser.add_argument(
        "--steady-shift",
        action="store_true",
        help="Enable aggressive pan direction during musically stable sections.",
    )
    parser.add_argument(
        "--steady-activation-mode",
        choices=["auto", "manual"],
        default="auto",
        help="auto=derive activation duration from music features, manual=use threshold+min-seconds.",
    )
    parser.add_argument(
        "--steady-activation-ratio",
        type=float,
        default=1.0,
        help="Auto mode: fraction of eligible music-driven shift runs to keep (0..1).",
    )
    parser.add_argument(
        "--steady-shift-probability",
        type=float,
        default=1.0,
        help="Auto mode: per-selected-run probability that shift is applied (0..1).",
    )
    parser.add_argument(
        "--steady-min-seconds",
        type=float,
        default=1.0,
        help="Manual mode only: minimum consecutive stable duration before shift activates.",
    )
    parser.add_argument(
        "--steady-threshold",
        type=float,
        default=0.72,
        help="Manual mode only: stability threshold in [0,1] for steady-shift activation.",
    )
    parser.add_argument(
        "--steady-shift-pixels",
        type=float,
        default=6.0,
        help="Base pan magnitude (pixels/frame) while steady-shift is active.",
    )
    parser.add_argument(
        "--steady-twist",
        action="store_true",
        help="Enable right-only twist (angle pulse) during steady music sections.",
    )
    parser.add_argument(
        "--steady-twist-max-deg",
        type=float,
        default=4.0,
        help="Maximum extra twist angle (degrees/frame) while steady-twist is active.",
    )
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument(
        "--init-image",
        type=str,
        default=None,
        help="Path to an initial image (jpg/png/webp). If set, frame 0 starts from this image.",
    )
    parser.add_argument(
        "--concept-mode",
        choices=["identity", "transform"],
        default="identity",
        help="How to treat init-image concept. identity=lock subject, transform=allow normal subject drift.",
    )
    parser.add_argument(
        "--identity-prompt",
        type=str,
        default="",
        help="Optional identity anchor text appended in identity concept mode.",
    )
    parser.add_argument(
        "--re-anchor",
        action="store_true",
        help="Periodically re-anchor diffusion to transformed frame 0 when --init-image is provided.",
    )
    parser.add_argument(
        "--re-anchor-strength",
        type=str,
        default="mid",
        help=(
            "Re-anchor strength preset (small|mid|big) or numeric blend alpha in [0,1] "
            "(e.g. 0.35)."
        ),
    )
    parser.add_argument(
        "--re-anchor-every-frames",
        type=int,
        default=12,
        help="Apply re-anchor every N diffusion frames.",
    )
    parser.add_argument(
        "--disable-camera-motion",
        action="store_true",
        help="Disable zoom/pan/rotation transform and keep camera static.",
    )
    parser.add_argument(
        "--disable-music-change",
        action="store_true",
        help="Disable music-driven control changes and use fixed controls each frame.",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help=(
            "Resume an interrupted run from this directory. "
            "The script loads frame_*.png and resume_state_<args_slug>.json. "
            "When resuming, --max-seconds is treated as additional duration and "
            "argument changes are allowed (reported as warnings)."
        ),
    )
    return parser


def main(argv: Optional[list] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    from .runner import run

    run(
        audio_path=args.audio,
        fps=args.fps,
        device=args.device,
        mode=args.mode,
        cadence=args.cadence,
        user_prompt=args.prompt,
        prompt_change_every_beats=args.prompt_change_every_beats,
        music_color_fx=args.music_color_fx,
        onset_jitter=args.onset_jitter,
        quiet_hold=args.quiet_hold,
        color_coherence_mode=args.color_coherence,
        color_coherence_strength=args.color_coherence_strength,
        prompt_mode=args.prompt_mode,
        coherence=args.coherence,
        subject_hold_frames=args.subject_hold_frames,
        subject_transition_frames=args.subject_transition_frames,
        subject_smoothing_window=args.subject_smoothing_window,
        max_seconds=args.max_seconds,
        with_audio=args.with_audio,
        use_controlnet=args.controlnet,
        controlnet_model=args.controlnet_model,
        control_scale_base=args.control_scale_base,
        control_scale_beat_boost=args.control_scale_beat_boost,
        control_scale_onset_boost=args.control_scale_onset_boost,
        zoom_base=args.zoom_base,
        zoom_beat_boost=args.zoom_beat_boost,
        steady_shift=args.steady_shift,
        steady_activation_mode=args.steady_activation_mode,
        steady_activation_ratio=args.steady_activation_ratio,
        steady_shift_probability=args.steady_shift_probability,
        steady_min_seconds=args.steady_min_seconds,
        steady_threshold=args.steady_threshold,
        steady_shift_pixels=args.steady_shift_pixels,
        steady_twist=args.steady_twist,
        steady_twist_max_deg=args.steady_twist_max_deg,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        init_image=args.init_image,
        concept_mode=args.concept_mode,
        identity_prompt=args.identity_prompt,
        re_anchor=args.re_anchor,
        re_anchor_strength=args.re_anchor_strength,
        re_anchor_every_frames=args.re_anchor_every_frames,
        disable_camera_motion=args.disable_camera_motion,
        disable_music_change=args.disable_music_change,
        resume_dir=args.resume_dir,
        cli_args=cli_args,
    )


if __name__ == "__main__":
    main()
