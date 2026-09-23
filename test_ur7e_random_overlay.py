"""
Standalone test: overlay a simulated UR7e onto an arbitrary hand-free video,
driven by a randomly generated end-effector trajectory (no hand tracking
required). Not part of the process_data.py pipeline -- run manually after
activating the `phantom` conda env (see install.sh).

This reuses the same TwinRobot / MujocoCameraParams machinery as
smoke_test_ur7e.py and the same real-camera calibration files (HD1080)
that phantom/processors/robotinpaint_processor.py uses for the Panda/UR7e
rig. Since the input video is arbitrary, frames are force-resized to that
calibration's resolution -- the overlay is best-effort, not guaranteed to
be geometrically exact unless the video was shot from the same rig.

Usage:
    python test_ur7e_random_overlay.py --video path/to/video.mp4 [options]
"""
import argparse
import json
import logging
import os

import cv2
import mediapy as media
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from phantom.twin_robot import TwinRobot, MujocoCameraParams, convert_real_camera_ori_to_mujoco

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKING_ERROR_THRESHOLD = 0.05  # meters, matches robotinpaint_processor.py

# Reachable workspace box used for random waypoints, matches the envelope
# spanned by smoke_test_ur7e.py's hand-picked TEST_POSES.
POS_LOW = np.array([0.35, -0.25, 0.25])
POS_HIGH = np.array([0.65, 0.25, 0.45])
GRIPPER_CHOICES = np.array([0.0, 0.04, 0.085])
BASE_ORI_XYZW = np.array([0, 1, 0, 0])  # gripper facing down


def build_real_camera_params(square: bool = True) -> MujocoCameraParams:
    """
    Mirrors RobotInpaintProcessor._get_mujoco_camera_params / smoke_test_ur7e.py's
    _build_real_camera_params, using the same real calibration files
    configs/ur7e.yaml points at.
    """
    with open(os.path.join(REPO_ROOT, "phantom/camera/camera_intrinsics_HD1080.json")) as f:
        intrinsics = json.load(f)["left"]
    with open(os.path.join(REPO_ROOT, "phantom/camera/camera_extrinsics.json")) as f:
        extrinsics = json.load(f)[0]

    img_w, img_h = 1080 * 16 // 9, 1080  # input_resolution=1080 (Phantom paper default)
    offset = (img_w - img_h) // 2 if square else 0
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"] + offset, intrinsics["cy"]
    sensor_width, sensor_height = img_w / fy / 1000, img_h / fx / 1000

    camera_ori_wxyz = convert_real_camera_ori_to_mujoco(np.array(extrinsics["camera_base_ori"]))

    return MujocoCameraParams(
        name="frontview",
        pos=np.array(extrinsics["camera_base_pos"]),
        ori_wxyz=camera_ori_wxyz,
        fov=intrinsics["v_fov"],
        resolution=(img_h, img_w),
        sensorsize=np.array([sensor_width, sensor_height]),
        principalpixel=np.array([img_w / 2 - cx, cy - img_h / 2]),
        focalpixel=np.array([fx, fy]),
    )


def generate_random_trajectory(num_frames: int, num_waypoints: int, seed: int):
    """
    Generate a smooth random end-effector trajectory across num_frames.

    Returns:
        positions (num_frames, 3), quats_xyzw (num_frames, 4), gripper_widths (num_frames,)
    """
    rng = np.random.default_rng(seed)

    waypoint_positions = rng.uniform(POS_LOW, POS_HIGH, size=(num_waypoints, 3))
    yaw_angles = rng.uniform(-60, 60, size=num_waypoints)
    base_rot = Rotation.from_quat(BASE_ORI_XYZW)
    waypoint_rotations = Rotation.concatenate(
        [base_rot * Rotation.from_euler("z", yaw, degrees=True) for yaw in yaw_angles]
    )
    waypoint_grippers = rng.choice(GRIPPER_CHOICES, size=num_waypoints)

    waypoint_frame_idxs = np.linspace(0, num_frames - 1, num_waypoints)
    frame_idxs = np.arange(num_frames)

    positions = np.stack(
        [np.interp(frame_idxs, waypoint_frame_idxs, waypoint_positions[:, i]) for i in range(3)],
        axis=1,
    )

    slerp = Slerp(waypoint_frame_idxs, waypoint_rotations)
    quats_xyzw = slerp(frame_idxs).as_quat()

    # Step function: hold each waypoint's gripper value until the next waypoint.
    gripper_widths = waypoint_grippers[np.searchsorted(waypoint_frame_idxs, frame_idxs, side="right") - 1]

    return positions, quats_xyzw, gripper_widths


def center_crop_square(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    n_remove = (width - height) // 2
    if n_remove <= 0:
        return img
    return img[:, n_remove:-n_remove]


def overlay_robot(frame: np.ndarray, robot_results: dict) -> np.ndarray:
    """Composite the rendered robot onto frame using its instance-segmentation masks."""
    rgb_sim = (robot_results["rgb_img"] * 255).astype(np.uint8)
    if rgb_sim.shape[:2] != frame.shape[:2]:
        rgb_sim = cv2.resize(rgb_sim, (frame.shape[1], frame.shape[0]))

    robot_mask = robot_results["robot_mask"]
    gripper_mask = robot_results["gripper_mask"]
    if robot_mask.shape[:2] != frame.shape[:2]:
        robot_mask = cv2.resize(robot_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
        gripper_mask = cv2.resize(gripper_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

    out = frame.copy()
    overlay_mask = (robot_mask > 0) | (gripper_mask > 0)
    out[overlay_mask] = rgb_sim[overlay_mask]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to input video (should contain no hands/arms)")
    parser.add_argument("--output", default=None, help="Output video path (default: <video_stem>_ur7e_overlay.mp4)")
    parser.add_argument("--robot", default="UR7e")
    parser.add_argument("--gripper", default="Robotiq85")
    parser.add_argument("--num-waypoints", type=int, default=5, help="Number of random trajectory waypoints")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random trajectory")
    parser.add_argument("--max-frames", type=int, default=150, help="Cap on number of frames to process")
    parser.add_argument("--square", action=argparse.BooleanOptionalAction, default=True,
                         help="Crop to square aspect ratio (matches configs/ur7e.yaml default)")
    parser.add_argument("--render", action="store_true", help="Open an on-screen MuJoCo viewer while stepping")
    args = parser.parse_args()

    output_path = args.output or (os.path.splitext(args.video)[0] + "_ur7e_overlay.mp4")

    camera_params = build_real_camera_params(square=args.square)
    img_h, img_w = camera_params.resolution

    print(f"Reading video: {args.video}")
    video = media.read_video(args.video)
    src_h, src_w = video.shape[1], video.shape[2]
    if abs((src_w / src_h) - (16 / 9)) > 0.1:
        print(f"WARNING: input video aspect ratio ({src_w}x{src_h}) differs from the calibrated 16:9 rig -- "
              "the overlay geometry will not be exact.")

    num_frames = min(len(video), args.max_frames)
    frames = [cv2.resize(np.asarray(video[i]), (img_w, img_h)) for i in range(num_frames)]
    if args.square:
        frames = [center_crop_square(f) for f in frames]

    print(f"Generating random trajectory: {num_frames} frames, {args.num_waypoints} waypoints, seed={args.seed}")
    positions, quats_xyzw, gripper_widths = generate_random_trajectory(num_frames, args.num_waypoints, args.seed)

    print(f"Instantiating TwinRobot(robot_name='{args.robot}', gripper_name='{args.gripper}')...")
    robot = TwinRobot(
        robot_name=args.robot,
        gripper_name=args.gripper,
        camera_params=camera_params,
        camera_height=img_h,
        camera_width=img_w,
        render=args.render,
        n_steps_short=3,
        n_steps_long=75,
        square=args.square,
    )
    print("TwinRobot initialized successfully.\n")

    output_frames = []
    num_skipped = 0
    try:
        for idx in range(num_frames):
            state = {"pos": positions[idx], "ori_xyzw": quats_xyzw[idx], "gripper_pos": gripper_widths[idx]}
            robot_results = robot.move_to_target_state(state, init=(idx == 0))

            if robot_results["pos_err"] > TRACKING_ERROR_THRESHOLD:
                logger.warning(f"Tracking error too large at frame {idx} ({robot_results['pos_err']:.4f} m), "
                                "skipping overlay for this frame")
                num_skipped += 1
                output_frames.append(frames[idx])
                continue

            output_frames.append(overlay_robot(frames[idx], robot_results))
    finally:
        robot.close()

    fps = getattr(video.metadata, "fps", None) or 15
    print(f"Writing {len(output_frames)} frames to {output_path} (fps={fps})")
    media.write_video(output_path, output_frames, fps=fps)

    print(f"\nDone. Processed {num_frames} frames, skipped {num_skipped} due to tracking error.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
