"""Video frame extraction for face detection."""

import math
from pathlib import Path

import numpy as np


def extract_frames(
    video_path: Path,
    max_duration: float = 15.0,
) -> list[np.ndarray] | None:
    """Extract frames from a video at 1 frame per second.

    Samples frames at 0, 1, 2, ..., floor(duration) seconds.
    Returns None if the video cannot be opened or exceeds max_duration.

    Args:
        video_path: Path to the video file.
        max_duration: Maximum allowed duration in seconds. Videos longer than this
            return None so the caller can handle skipping.

    Returns:
        List of BGR numpy arrays (one per sampled frame), or None if the file
        cannot be opened or duration exceeds max_duration.
    """
    import cv2  # noqa: PLC0415 — deferred to avoid import cost when cv2 is not installed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Guard against invalid metadata (some containers report 0 or -1)
        if video_fps <= 0 or total_frames <= 0:
            return None

        duration = total_frames / video_fps

        if duration > max_duration:
            return None

        frames: list[np.ndarray] = []
        # Sample at seconds 0, 1, 2, ..., floor(duration)
        sample_seconds = range(0, math.floor(duration) + 1)

        for second in sample_seconds:
            frame_index = int(second * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        return frames if frames else None
    finally:
        cap.release()
