"""
Standalone test: project the recorded end-effector trajectory
(pick_and_place/trajectoryPushT.json) onto pick_and_place/0/pushT.mp4 using
the new hand-eye calibration (OpenCV-style base->camera extrinsics) and the
episode's own cam_intrinsics.json.

Trajectory positions are in the robot base frame and are timestamped at
~10 Hz; pushT.mp4 is 30 fps and its duration (42.7s) matches
num_timesteps * dt (426 * 0.1 = 42.6s), so trajectory time 0 is assumed to
align with video frame 0 (t - t[0] == video seconds).

This is a one-off sanity test, only rendering the [--start, --end) window
(default 12s-20s) rather than the whole video.

Usage:
    python test_pushT_trajectory_overlay.py [--start 12] [--end 20]
"""
import argparse
import json
import os

import cv2
import mediapy as media
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EPISODE_DIR = os.path.join(REPO_ROOT, "pick_and_place", "0")

# OpenCV-style extrinsics: base -> camera (X_cam = R_cam_base @ X_base + t_cam_base).
# The raw hand-eye result (bTc) gives ~162px reprojection error against the
# tracked pusher tip even with correct 640->1280 scaling (a real ~15cm/10deg
# discrepancy, not a scaling bug -- likely a reference-frame offset between
# the checkerboard calibration's "base" and the UR's own ActualTCPPose). So
# this is bTc refined via solvePnP against the tracked pusher tip (same
# 240-point correspondences as before), using the hand-eye intrinsics/distortion
# below. Converges to ~13.8px, essentially the same pose as the earlier
# from-scratch video fit.
R_CAM_BASE = np.array([
    [-0.05448752007510815, 0.9916828927741412, -0.1166025314269646],
    [0.9293049050351305, 0.007645458091250168, -0.3692342622891629],
    [-0.3652718215708845, -0.12847796367348968, -0.9219923585456669],
])
T_CAM_BASE = np.array([0.2929124119034776, -0.2927471892253993, 0.6887606058268763])

# Hand-eye calibration's own checkerboard intrinsics, done at 640x400.
CALIB_CAMERA_MATRIX = np.array([
    [412.5838771540844, 0.0, 340.793428357155],
    [0.0, 410.7020163307825, 186.2750819311434],
    [0.0, 0.0, 1.0],
])
CALIB_DIST_COEFS = np.array([
    0.18674622221871978, -0.7733606887634911, -0.0033474655508258957, 0.0018426852308953648, 0.916631780513466
])
CALIB_RESOLUTION = (640, 400)  # (width, height)

TRAJECTORY_HZ = 10.0


def load_trajectory():
    with open(os.path.join(REPO_ROOT, "pick_and_place", "trajectoryPushT.json")) as f:
        traj = json.load(f)
    t = np.array([p["t"] for p in traj["positions"]])
    pos = np.array([p["position"] for p in traj["positions"]])
    rel_t = t - t[0]
    return rel_t, pos


def load_intrinsics(video_w: int, video_h: int):
    # CALIB_CAMERA_MATRIX was calibrated at 640x400; pushT.mp4 is 1280x720 --
    # different aspect ratio, so scale each axis independently rather than
    # assuming a uniform/centered crop.
    calib_w, calib_h = CALIB_RESOLUTION
    sx, sy = video_w / calib_w, video_h / calib_h

    fx = CALIB_CAMERA_MATRIX[0, 0] * sx
    fy = CALIB_CAMERA_MATRIX[1, 1] * sy
    cx = CALIB_CAMERA_MATRIX[0, 2] * sx
    cy = CALIB_CAMERA_MATRIX[1, 2] * sy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def project_points(points_base: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """points_base: (N, 3) points in robot base frame -> (N, 2) pixel coords."""
    rvec, _ = cv2.Rodrigues(R_CAM_BASE)
    tvec = T_CAM_BASE.reshape(3, 1)
    pixels, _ = cv2.projectPoints(points_base, rvec, tvec, camera_matrix, distCoeffs=CALIB_DIST_COEFS)
    return pixels.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=float, default=12.0, help="Start time in the video (s)")
    parser.add_argument("--end", type=float, default=20.0, help="End time in the video (s)")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--trail", type=int, default=30, help="Number of past frames to draw as a fading trail")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        EPISODE_DIR, f"pushT_traj_overlay_{int(args.start)}s_{int(args.end)}s.mp4"
    )

    rel_t, positions_base = load_trajectory()

    print(f"Reading video: {os.path.join(EPISODE_DIR, 'pushT.mp4')}")
    video = media.read_video(os.path.join(EPISODE_DIR, "pushT.mp4"))
    fps = getattr(video.metadata, "fps", None) or 30.0
    video_h, video_w = video.shape[1], video.shape[2]

    camera_matrix = load_intrinsics(video_w, video_h)
    print(f"Video: {video_w}x{video_h} @ {fps}fps, scaled camera_matrix:\n{camera_matrix}")

    start_frame = int(round(args.start * fps))
    end_frame = min(int(round(args.end * fps)), len(video))
    print(f"Rendering frames [{start_frame}, {end_frame}) -> {end_frame - start_frame} frames "
          f"({args.start}s - {args.end}s)")

    # Interpolate the 10Hz trajectory to every video frame time in the window.
    frame_idxs = np.arange(start_frame, end_frame)
    frame_times = frame_idxs / fps
    interp_positions = np.stack(
        [np.interp(frame_times, rel_t, positions_base[:, i]) for i in range(3)], axis=1
    )
    pixels = project_points(interp_positions, camera_matrix)

    # Also project the full trajectory once for the fading trail buffer.
    all_pixels = project_points(positions_base, camera_matrix)

    output_frames = []
    for k, frame_idx in enumerate(frame_idxs):
        frame = np.asarray(video[frame_idx]).copy()
        t_now = frame_times[k]

        # Fading trail: prior trajectory samples within the last `trail` samples in time.
        trail_mask = (rel_t <= t_now) & (rel_t > t_now - args.trail / TRAJECTORY_HZ)
        trail_px = all_pixels[trail_mask]
        for i, (x, y) in enumerate(trail_px):
            if 0 <= x < video_w and 0 <= y < video_h:
                alpha = (i + 1) / max(len(trail_px), 1)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        x, y = pixels[k]
        if 0 <= x < video_w and 0 <= y < video_h:
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
        else:
            # Off-frame: clamp and draw at the edge so the direction is still visible.
            cx_clamped = int(np.clip(x, 5, video_w - 5))
            cy_clamped = int(np.clip(y, 5, video_h - 5))
            cv2.drawMarker(frame, (cx_clamped, cy_clamped), (0, 0, 255),
                            markerType=cv2.MARKER_TRIANGLE_DOWN if y >= video_h else cv2.MARKER_TRIANGLE_UP,
                            markerSize=16, thickness=2)
            print(f"  WARNING: frame {frame_idx} (t={t_now:.2f}s) projects outside image bounds: ({x:.1f}, {y:.1f})")

        cv2.putText(frame, f"t={t_now:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        output_frames.append(frame)

    print(f"Writing {len(output_frames)} frames to {output_path} (fps={fps})")
    media.write_video(output_path, output_frames, fps=fps)
    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
