#!/usr/bin/env python3
"""Extract a random frame from a video using ffmpeg/ffprobe."""

# python3 extract_random_frame.py "/workspaces/pose_randomization_inpainting/pick_and_place/Empty_workspace.mp4" "/workspaces/pose_randomization_inpainting/pick_and_place/Empty_workspace.png"

import argparse
import random
import subprocess
from pathlib import Path


def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path, nargs="?")
    args = parser.parse_args()

    if not args.video_path.is_file():
        parser.error(f"video file not found: {args.video_path}")

    duration = get_duration(args.video_path)
    margin = min(0.5, duration * 0.01)
    timestamp = random.uniform(0, max(duration - margin, 0))

    output_path = args.output_path
    if output_path is None:
        safe_ts = f"{timestamp:.3f}".replace(".", "_")
        output_path = args.video_path.with_name(
            f"{args.video_path.stem}_frame_{safe_ts}.png"
        )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", str(args.video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
            "-loglevel", "error",
        ],
        check=True,
    )

    print(f"Extracted frame at t={timestamp:.3f}s -> {output_path}")


if __name__ == "__main__":
    main()
