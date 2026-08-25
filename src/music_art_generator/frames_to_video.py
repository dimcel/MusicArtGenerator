"""
Simple Frames to Video Converter

Reads frames from a directory and creates a video using ffmpeg.
Useful for testing and manual video creation.

Usage:
    python -m music_art_generator.frames_to_video --frames_dir output_frames --output video.mp4
    python -m music_art_generator.frames_to_video --frames_dir my_test --fps 30
"""

import os
import argparse
from pathlib import Path


def frames_to_video(
    frames_dir: str,
    output_path: str = None,
    fps: int = 24,
    pattern: str = None
):
    """
    Convert frames to video using ffmpeg.
    Handles gaps in frame sequences automatically.
    
    Args:
        frames_dir: Directory containing frames
        output_path: Output video path (default: frames_dir/video.mp4)
        fps: Frames per second
        pattern: Optional glob pattern (e.g., "*.png", "frame_*.jpg")
                 If None, finds all image files automatically
    
    Returns:
        Path to created video
    """
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        print(f"❌ Error: Directory '{frames_dir}' not found")
        return None
    
    # Find all image files
    if pattern:
        frames = sorted(frames_dir.glob(pattern))
    else:
        # Find all common image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        frames = []
        for ext in extensions:
            frames.extend(frames_dir.glob(ext))
        frames = sorted(frames)
    
    if not frames:
        print(f"❌ Error: No frames found in '{frames_dir}'")
        if pattern:
            print(f"   Pattern used: {pattern}")
        return None
    
    print(f"\n🎬 Converting frames to video...")
    print(f"   Input: {frames_dir}/")
    print(f"   Found: {len(frames)} frames")
    print(f"   First: {frames[0].name}")
    print(f"   Last: {frames[-1].name}")
    print(f"   FPS: {fps}")
    
    # Default output path
    if output_path is None:
        output_path = frames_dir / "video.mp4"
    
    # Create temporary file list for ffmpeg concat demuxer
    # This handles gaps in frame sequences
    filelist_path = frames_dir / "ffmpeg_filelist.txt"
    
    try:
        with open(filelist_path, 'w') as f:
            for frame in frames:
                # Write absolute path with proper escaping
                f.write(f"file '{frame.absolute()}'\n")
        
        print(f"   Output: {output_path}")
        print(f"\n   Running ffmpeg (using concat demuxer for gap handling)...")
        
        # ffmpeg command using concat demuxer
        cmd = (
            f'ffmpeg -y -f concat -safe 0 -r {fps} '
            f'-i "{filelist_path}" '
            f'-c:v libx264 -pix_fmt yuv420p '
            f'-crf 23 '
            f'"{output_path}"'
        )
        
        result = os.system(cmd)
        
        # Clean up temporary file
        filelist_path.unlink()
        
        if result == 0:
            print(f"\n   ✓ Video created successfully!")
            print(f"   Location: {output_path}")
            print(f"   Frames used: {len(frames)}")
            print(f"   Duration: {len(frames) / fps:.2f} seconds")
            return str(output_path)
        else:
            print(f"\n   ❌ Failed to create video (ffmpeg error)")
            return None
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        if filelist_path.exists():
            filelist_path.unlink()
        return None


def list_frames(frames_dir: str):
    """List all frame files in directory"""
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        print(f"❌ Directory '{frames_dir}' not found")
        return
    
    # Find all image files
    extensions = ['.png', '.jpg', '.jpeg']
    frames = []
    
    for ext in extensions:
        frames.extend(sorted(frames_dir.glob(f'*{ext}')))
    
    if not frames:
        print(f"❌ No frames found in '{frames_dir}'")
        return
    
    print(f"\n📁 Found {len(frames)} frames in '{frames_dir}':")
    for i, frame in enumerate(frames[:10], 1):
        print(f"   {i}. {frame.name}")
    
    if len(frames) > 10:
        print(f"   ... and {len(frames) - 10} more")
    
    print(f"\nFirst frame: {frames[0].name}")
    print(f"Last frame:  {frames[-1].name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert frames to video using ffmpeg"
    )
    
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing frames"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: frames_dir/video.mp4)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional glob pattern (e.g., '*.png', 'frame_*.jpg'). If not specified, finds all images."
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List frames in directory without creating video"
    )
    
    args = parser.parse_args()
    
    if args.list:
        # Just list frames
        list_frames(args.frames_dir)
    else:
        # Create video
        video_path = frames_to_video(
            frames_dir=args.frames_dir,
            output_path=args.output,
            fps=args.fps,
            pattern=args.pattern
        )
        
        if video_path:
            print("\n✓ Done!")


if __name__ == "__main__":
    main()
