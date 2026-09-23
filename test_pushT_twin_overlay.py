"""
Standalone test: composite the simulated UR7e twin onto pushT.mp4, driven by
the REAL recorded end-effector trajectory (pick_and_place/trajectoryPushT.json,
UR "ActualTCPPose") instead of a random one, using the camera calibration
fitted against this exact episode (see test_pushT_trajectory_overlay.py).

Only position is recorded in trajectoryPushT.json, so orientation is held
fixed (gripper facing straight down, matching the pusher tool seen in the
video) and the gripper is held closed to approximate the rigid pusher.

Like test_ur7e_random_overlay.py, this reuses the TwinRobot / MujocoCameraParams
machinery but points it at this episode's own camera_intrinsics (scaled to
pushT.mp4's native 1280x720 -- no cropping needed since the extrinsics were
fit directly at that resolution).

Usage:
    python test_pushT_twin_overlay.py [--start 12] [--end 20]
"""
import argparse
import json
import logging
import os

import cv2
import mediapy as media
import numpy as np

from phantom.twin_robot import TwinRobot, MujocoCameraParams, convert_real_camera_ori_to_mujoco

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EPISODE_DIR = os.path.join(REPO_ROOT, "pick_and_place", "0")
TRACKING_ERROR_THRESHOLD = 0.05  # meters, matches robotinpaint_processor.py

# Pusher tool held facing straight down, fixed for the whole episode (only
# position is recorded in trajectoryPushT.json).
FIXED_ORI_XYZW = np.array([0, 1, 0, 0])
GRIPPER_CLOSED = 0.0


# Hand-eye calibration's own checkerboard intrinsics, done at 640x400.
CALIB_CAMERA_MATRIX = np.array([
    [412.5838771540844, 0.0, 340.793428357155],
    [0.0, 410.7020163307825, 186.2750819311434],
    [0.0, 0.0, 1.0],
])
CALIB_RESOLUTION = (640, 400)  # (width, height)


def build_camera_params(video_w: int, video_h: int) -> MujocoCameraParams:
    with open(os.path.join(REPO_ROOT, "phantom/camera/camera_extrinsics.json")) as f:
        extrinsics = json.load(f)[0]

    # CALIB_CAMERA_MATRIX was calibrated at 640x400; pushT.mp4 is 1280x720 --
    # different aspect ratio, so scale each axis independently rather than
    # assuming a uniform/centered crop.
    calib_w, calib_h = CALIB_RESOLUTION
    sx, sy = video_w / calib_w, video_h / calib_h
    fx = CALIB_CAMERA_MATRIX[0, 0] * sx
    fy = CALIB_CAMERA_MATRIX[1, 1] * sy
    cx = CALIB_CAMERA_MATRIX[0, 2] * sx
    cy = CALIB_CAMERA_MATRIX[1, 2] * sy
    sensor_width, sensor_height = video_w / fy / 1000, video_h / fx / 1000

    # MuJoCo's pinhole camera has no lens-distortion model, so this ignores
    # CALIB_DIST_COEFS -- the composited robot mask will be geometrically
    # approximate near the frame edges where distortion is largest.
    v_fov = np.degrees(2 * np.arctan(video_h / (2 * fy)))

    camera_ori_wxyz = convert_real_camera_ori_to_mujoco(np.array(extrinsics["camera_base_ori"]))

    return MujocoCameraParams(
        name="frontview",
        pos=np.array(extrinsics["camera_base_pos"]),
        ori_wxyz=camera_ori_wxyz,
        fov=v_fov,
        resolution=(video_h, video_w),
        sensorsize=np.array([sensor_width, sensor_height]),
        principalpixel=np.array([video_w / 2 - cx, cy - video_h / 2]),
        focalpixel=np.array([fx, fy]),
    )


def load_trajectory():
    with open(os.path.join(REPO_ROOT, "pick_and_place", "trajectoryPushT.json")) as f:
        traj = json.load(f)
    t = np.array([p["t"] for p in traj["positions"]])
    pos = np.array([p["position"] for p in traj["positions"]])
    return t - t[0], pos


def overlay_robot(frame: np.ndarray, robot_results: dict) -> np.ndarray:
    rgb_sim = (robot_results["rgb_img"] * 255).astype(np.uint8)
    robot_mask = robot_results["robot_mask"]
    gripper_mask = robot_results["gripper_mask"]

    out = frame.copy()
    overlay_mask = (robot_mask > 0) | (gripper_mask > 0)
    out[overlay_mask] = rgb_sim[overlay_mask]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=float, default=12.0, help="Start time in the video (s)")
    parser.add_argument("--end", type=float, default=20.0, help="End time in the video (s)")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--render", action="store_true", help="Open an on-screen MuJoCo viewer while stepping")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        EPISODE_DIR, f"pushT_twin_overlay_{int(args.start)}s_{int(args.end)}s.mp4"
    )

    rel_t, positions_base = load_trajectory()

    print(f"Reading video: {os.path.join(EPISODE_DIR, 'pushT.mp4')}")
    video = media.read_video(os.path.join(EPISODE_DIR, "pushT.mp4"))
    fps = getattr(video.metadata, "fps", None) or 30.0
    video_h, video_w = video.shape[1], video.shape[2]

    camera_params = build_camera_params(video_w, video_h)
    print(f"Camera params: pos={camera_params.pos}, resolution={camera_params.resolution}")

    start_frame = int(round(args.start * fps))
    end_frame = min(int(round(args.end * fps)), len(video))
    frame_idxs = np.arange(start_frame, end_frame)
    frame_times = frame_idxs / fps
    interp_positions = np.stack(
        [np.interp(frame_times, rel_t, positions_base[:, i]) for i in range(3)], axis=1
    )
    print(f"Driving TwinRobot over frames [{start_frame}, {end_frame}) "
          f"({args.start}s - {args.end}s, {len(frame_idxs)} frames)")

    print("Instantiating TwinRobot(robot_name='UR7e', gripper_name='Robotiq85')...")
    robot = TwinRobot(
        robot_name="UR7e",
        gripper_name="Robotiq85",
        camera_params=camera_params,
        camera_height=video_h,
        camera_width=video_w,
        render=args.render,
        n_steps_short=3,
        n_steps_long=75,
        square=False,
    )
    print("TwinRobot initialized successfully.\n")

    output_frames = []
    num_skipped = 0
    try:
        for k, frame_idx in enumerate(frame_idxs):
            frame = np.asarray(video[frame_idx])
            state = {"pos": interp_positions[k], "ori_xyzw": FIXED_ORI_XYZW, "gripper_pos": GRIPPER_CLOSED}
            robot_results = robot.move_to_target_state(state, init=(k == 0))

            if robot_results["pos_err"] > TRACKING_ERROR_THRESHOLD:
                logger.warning(f"Tracking error too large at frame {frame_idx} "
                                f"({robot_results['pos_err']:.4f} m), skipping overlay for this frame")
                num_skipped += 1
                output_frames.append(frame)
                continue

            output_frames.append(overlay_robot(frame, robot_results))
    finally:
        robot.close()

    print(f"Writing {len(output_frames)} frames to {output_path} (fps={fps})")
    media.write_video(output_path, output_frames, fps=fps)
    print(f"\nDone. Processed {len(frame_idxs)} frames, skipped {num_skipped} due to tracking error.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
