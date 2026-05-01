"""Recursive video file discovery."""

import os
from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def scan_videos(inputs: list[Path]) -> list[Path]:
    """Scan inputs for supported video files.

    For directories, recurse into them without following symlinks (to avoid
    infinite loops from circular symlink structures). For individual files, use
    them directly. Returns a sorted list of unique video paths.

    Args:
        inputs: List of Path objects (files or directories).

    Returns:
        Sorted list of unique video paths with supported extensions.
    """
    found: set[Path] = set()

    for input_path in inputs:
        if input_path.is_dir():
            for root, _dirs, files in os.walk(input_path, followlinks=False):
                for fname in files:
                    p = Path(root) / fname
                    if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                        found.add(p.resolve())
        elif input_path.is_file():
            if input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                found.add(input_path.resolve())

    return sorted(found)
