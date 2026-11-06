"""
Shared IO + geometry for the video-based planar mosaic pipeline.

Conventions
-----------
- Robot poses: robot_eef_pose = [x, y, z, rx, ry, rz] (meters, rotation vector),
  i.e. T_base_eef.
- Hand-eye calibration: camera_extrinsics.json {"fTc": 4x4}, flange -> camera,
  translation in millimeters.
- Camera pose of a frame: T_base_cam = T_base_eef @ T_eef_cam. Camera frame is the
  OpenCV optical frame (x right, y down, z forward).
- Work surface: plane n . X = d in the robot base frame.
- Mosaic canvas: plane point X = origin + s*u + t*v; canvas pixel
  (col, row) = ((s - s_min), (t - t_min)) * px_per_m.
"""

import json
import pathlib

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp


# ----------------------------------------------------------------------------- IO

def load_episode(zarr_path, episode_idx):
    """Return dict of numpy arrays for one episode of a diffusion_policy replay buffer."""
    root = zarr.open(str(zarr_path), mode='r')
    ends = root['meta/episode_ends'][:]
    if episode_idx < 0:
        episode_idx += len(ends)
    start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
    end = int(ends[episode_idx])
    return {k: v[start:end] for k, v in root['data'].items()}, episode_idx


def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    K = np.array(d['camera_matrix'], dtype=np.float64)
    dist = np.array(d.get('dist_coefs', [[0, 0, 0, 0, 0]]), dtype=np.float64).ravel()
    return K, dist


def load_extrinsics(path):
    with open(path) as f:
        d = json.load(f)
    T = np.array(d['fTc'], dtype=np.float64)
    T[:3, 3] /= 1000.0  # mm -> m
    return T


def iter_video(path, indices=None):
    """Yield (frame_idx, BGR frame). If `indices` is given, only those frames are
    decoded into images (others are grabbed and skipped)."""
    cap = cv2.VideoCapture(str(path))
    wanted = None if indices is None else set(int(i) for i in indices)
    last = None if indices is None else max(wanted)
    i = 0
    while True:
        if wanted is not None and i > last:
            break
        if wanted is None or i in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


# ----------------------------------------------------------------------- poses

def pose6_to_matrix(pose6):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(pose6[3:6]).as_matrix()
    T[:3, 3] = pose6[:3]
    return T


def matrix_to_pose6(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


class PoseInterpolator:
    """Continuous-time eef pose: linear in translation, slerp in rotation.
    (Rotation vectors near pi flip sign between samples in this data; going through
    Rotation objects makes that harmless.)"""

    def __init__(self, timestamps, poses6):
        self.t = np.asarray(timestamps, dtype=np.float64)
        self.p = np.asarray(poses6[:, :3], dtype=np.float64)
        self.slerp = Slerp(self.t, Rotation.from_rotvec(poses6[:, 3:6]))

    def __call__(self, times):
        times = np.clip(np.asarray(times, dtype=np.float64), self.t[0], self.t[-1])
        pos = np.stack([np.interp(times, self.t, self.p[:, k]) for k in range(3)], axis=-1)
        rot = self.slerp(times).as_matrix()
        T = np.tile(np.eye(4), (len(times), 1, 1))
        T[:, :3, :3] = rot
        T[:, :3, 3] = pos
        return T


# ----------------------------------------------------------------------- plane

class Plane:
    """n . X = d in base frame, with an in-plane orthonormal basis (u, v)."""

    def __init__(self, normal, d):
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        self.n, self.d = n, float(d)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        # u ~ base +y, v ~ base -x for a horizontal plane (top-down map orientation)
        u = np.cross(n, ref)
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(n, self.u)
        self.origin = n * self.d

    @classmethod
    def from_height_tilt(cls, z, tilt_x=0.0, tilt_y=0.0):
        """Plane through (0,0,z) whose normal is +z rotated by tilt_x about x and
        tilt_y about y (radians)."""
        n = Rotation.from_euler('xy', [tilt_x, tilt_y]).apply([0.0, 0.0, 1.0])
        return cls(n, n @ np.array([0.0, 0.0, z]))

    def intersect(self, origins, dirs):
        """Ray/plane intersection. origins (..,3) dirs (..,3) -> points (..,3), depth t."""
        denom = dirs @ self.n
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        t = (self.d - origins @ self.n) / denom
        return origins + t[..., None] * dirs, t

    def to_st(self, X):
        rel = X - self.origin
        return np.stack([rel @ self.u, rel @ self.v], axis=-1)

    def to_dict(self):
        return {'normal': self.n.tolist(), 'd': self.d}


class Canvas:
    """Pixel grid on the plane."""

    def __init__(self, plane, s_min, t_min, width, height, px_per_m):
        self.plane = plane
        self.s_min, self.t_min = float(s_min), float(t_min)
        self.width, self.height = int(width), int(height)
        self.res = float(px_per_m)

    @classmethod
    def covering(cls, plane, st_points, px_per_m, margin):
        lo = st_points.min(axis=0) - margin
        hi = st_points.max(axis=0) + margin
        w = int(np.ceil((hi[0] - lo[0]) * px_per_m))
        h = int(np.ceil((hi[1] - lo[1]) * px_per_m))
        return cls(plane, lo[0], lo[1], w, h, px_per_m)

    def H_canvas_to_base(self):
        """4x3 matrix A with X_base_h = A @ [col, row, 1]."""
        p = self.plane
        O = p.origin + self.s_min * p.u + self.t_min * p.v
        A = np.zeros((4, 3))
        A[:3, 0] = p.u / self.res
        A[:3, 1] = p.v / self.res
        A[:3, 2] = O
        A[3, 2] = 1.0
        return A

    def homography_canvas_to_image(self, K, T_base_cam):
        """3x3 H with (pinhole, undistorted) image pixel ~ H @ [col, row, 1]."""
        T_cam_base = np.linalg.inv(T_base_cam)
        P = K @ T_cam_base[:3, :]
        return P @ self.H_canvas_to_base()

    def st_to_px(self, st):
        return np.stack([(st[..., 0] - self.s_min) * self.res,
                         (st[..., 1] - self.t_min) * self.res], axis=-1)

    def to_dict(self):
        return {'plane': self.plane.to_dict(), 's_min': self.s_min, 't_min': self.t_min,
                'width': self.width, 'height': self.height, 'px_per_m': self.res}

    @classmethod
    def from_dict(cls, d):
        plane = Plane(d['plane']['normal'], d['plane']['d'])
        return cls(plane, d['s_min'], d['t_min'], d['width'], d['height'], d['px_per_m'])


# ------------------------------------------------------------------- camera

def pixel_rays(K, dist, w, h, step=1):
    """Unit-z camera-frame ray directions (h', w', 3) for every `step`-th pixel of the
    *raw (distorted)* image."""
    xs, ys = np.meshgrid(np.arange(0, w, step, dtype=np.float64),
                         np.arange(0, h, step, dtype=np.float64))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(ys.shape + (2,))
    return np.concatenate([und, np.ones(ys.shape + (1,))], axis=-1)


def footprint_st(plane, K, dist, w, h, T_base_cam, n_edge=16):
    """Plane (s,t) coords of the image border's ground footprint."""
    e = np.linspace(0, 1, n_edge)
    border = np.concatenate([
        np.stack([e * (w - 1), np.zeros_like(e)], 1),
        np.stack([np.full_like(e, w - 1), e * (h - 1)], 1),
        np.stack([e[::-1] * (w - 1), np.full_like(e, h - 1)], 1),
        np.stack([np.zeros_like(e), e[::-1] * (h - 1)], 1)])
    und = cv2.undistortPoints(border.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays_c = np.concatenate([und, np.ones((len(und), 1))], 1)
    R, c = T_base_cam[:3, :3], T_base_cam[:3, 3]
    X, _ = plane.intersect(np.broadcast_to(c, rays_c.shape), rays_c @ R.T)
    return plane.to_st(X)


def undistort_maps(K, dist, w, h):
    return cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
