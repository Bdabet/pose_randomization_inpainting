"""Clean-plate background loading and per-frame background sampling."""
from typing import List

import cv2
import mediapy as media
import numpy as np

from phantom.utils.image_utils import center_crop_square


def load_background_frames(video_path: str, max_frames: int, img_w: int, img_h: int, square: bool) -> List[np.ndarray]:
    """Read up to max_frames of a clean-plate video, resized to (img_w, img_h) and optionally square-cropped."""
    video = media.read_video(video_path)
    num_frames = min(len(video), max_frames)
    frames = [cv2.resize(np.asarray(video[i]), (img_w, img_h)) for i in range(num_frames)]
    if square:
        frames = [center_crop_square(f) for f in frames]
    return frames


def sample_background_indices(num_frames_out: int, num_bg_frames: int, hold_min: int, hold_max: int,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Independent-random background sampling, but each randomly chosen frame is
    held for a few output frames instead of re-rolled every single frame.
    Keeps the "random frame from the mostly-static video" behavior while
    avoiding single-frame flicker from sensor noise/auto-exposure.
    """
    idxs = []
    while len(idxs) < num_frames_out:
        idx = int(rng.integers(0, num_bg_frames))
        hold = int(rng.integers(hold_min, hold_max + 1))
        idxs.extend([idx] * hold)
    return np.array(idxs[:num_frames_out])
