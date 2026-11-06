import json
import numpy as np
import cv2
import os
import mediapy as media
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy.spatial.transform import Rotation

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )

def normalize_extrinsics(camera_extrinsics) -> List[Dict]:
    """
    Accept either calibration format and return the [{camera_base_pos,
    camera_base_ori}] one the rest of the code expects:

    - [{"camera_base_pos": [...], "camera_base_ori": 3x3}]  (as written by the
      marker-based calibration)
    - {"xyz": [...], "rpy": [...], "degrees": bool}         (hand-written pose)

    Both describe the same thing: camera position in the robot base frame and the
    camera's orientation in the OpenCV optical convention (x right, y down,
    z forward).
    """
    if isinstance(camera_extrinsics, dict) and "xyz" in camera_extrinsics:
        ori = Rotation.from_euler("xyz", camera_extrinsics["rpy"],
                                  degrees=camera_extrinsics.get("degrees", True)).as_matrix()
        return [{"camera_base_pos": list(camera_extrinsics["xyz"]),
                 "camera_base_ori": ori.tolist()}]
    return camera_extrinsics


def get_transformation_matrix_from_extrinsics(camera_extrinsics) -> np.ndarray:
    """Get homogeneous transformation matrix from camera extrinsics (either format)."""
    camera_extrinsics = normalize_extrinsics(camera_extrinsics)
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)
    return T_cam2robot


def get_intrinsics_from_json(json_path: str, target_resolution: Optional[Tuple[int, int]] = None
                             ) -> Tuple[np.ndarray, dict]:
    """
    Load camera intrinsics from either calibration format:

    - {"left": {fx, fy, cx, cy, v_fov, ...}}         (ZED SDK dump; already in pixels
      of the resolution it was dumped at, and it does not record that resolution)
    - {"camera_matrix": 3x3, "dist_coefs": [...],    (OpenCV calibrateCamera output)
       "resolution": [w, h]}

    `target_resolution` is the (w, h) the pipeline actually processes at. When the file
    records its own calibration resolution and the two differ, the intrinsics are
    scaled to it -- a 1280x720 calibration used on 1920x1080 frames is otherwise off by
    the 1.5x in focal length and principal point.
    """
    with open(json_path, "r") as f:
        camera_intrinsics = json.load(f)

    dist = np.zeros(5)
    if "left" in camera_intrinsics:
        p = camera_intrinsics["left"]
        fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
        v_fov, calib_wh = p.get("v_fov"), None
    elif "camera_matrix" in camera_intrinsics:
        m = np.array(camera_intrinsics["camera_matrix"], dtype=float)
        fx, fy, cx, cy = m[0, 0], m[1, 1], m[0, 2], m[1, 2]
        dist = np.array(camera_intrinsics.get("dist_coefs", [[0, 0, 0, 0, 0]]), dtype=float).ravel()
        v_fov = None
        calib_wh = tuple(camera_intrinsics["resolution"]) if "resolution" in camera_intrinsics else None
    else:
        raise ValueError(
            f"Unrecognised camera intrinsics format in {json_path}: expected a 'left' "
            "block (ZED) or a 'camera_matrix' (OpenCV)"
        )

    if target_resolution is not None and calib_wh is not None:
        sx, sy = target_resolution[0] / calib_wh[0], target_resolution[1] / calib_wh[1]
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy

    if v_fov is None:
        img_h = target_resolution[1] if target_resolution is not None else calib_wh[1]
        v_fov = float(np.degrees(2 * np.arctan(img_h / (2 * fy))))

    intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "v_fov": v_fov, "dist": dist}
    return intrinsics_matrix, intrinsics_dict

def resize_binary_image(image: np.ndarray, new_size: int) -> np.ndarray:
    max_value = np.max(image)

    # Resize the image
    resized_image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

    if max_value == 1:
        _, binary_image = cv2.threshold(resized_image, 0.5, 1, cv2.THRESH_BINARY)
    else:
        _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    return binary_image


def convert_video_to_images(video_path: str, save_folder: str, square=False, reverse=False):
    """Save each frame of video as an image in save_folder."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    imgs = np.array(media.read_video(str(video_path)))
    n_imgs = len(imgs)
    if reverse:
        imgs = imgs[::-1]
    for idx in range(n_imgs):
        img = imgs[idx]
        if square:
            delta = (img.shape[1] - img.shape[0]) // 2
            img = img[:, delta:-delta, :]
        media.write_image(f"{save_folder}/{idx:05d}.jpg", img)


