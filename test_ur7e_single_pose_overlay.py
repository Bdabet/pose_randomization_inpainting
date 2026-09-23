"""
Standalone test: overlay a simulated UR7e in a single fixed pose onto a
single arbitrary hand-free photo (no trajectory, no video). Not part of the
process_data.py pipeline -- run manually after activating the `phantom`
conda env (see install.sh).

This is the single-pose counterpart to test_ur7e_random_overlay.py: it
reuses the same TwinRobot / MujocoCameraParams machinery, but instead of
driving the robot along a trajectory across video frames, it settles the
robot into one target pose and composites it onto one image. The photo's
own native resolution is used for rendering (no forced resize/crop), so
the camera calibration is scaled to match it, the same way
test_pushT_twin_overlay.py scales its checkerboard calibration to
pushT.mp4's resolution.

Camera extrinsics and intrinsics are each supplied as a separate JSON file:

    extrinsics (xyz position + rpy orientation, degrees by default):
        {
            "xyz": [0.5396, -0.1997, 0.5611],
            "rpy": [-142.7, -18.0, -160.5],
            "degrees": true
        }

    intrinsics (OpenCV-style camera matrix + distortion coefficients,
    "resolution" is the [width, height] the calibration was done at --
    the matrix is scaled from this to the photo's actual resolution;
    dist_coefs is accepted but unused since MuJoCo's pinhole camera has
    no lens-distortion model):
        {
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist_coefs": [[k1, k2, p1, p2, k3]],
            "resolution": [640, 400]
        }

See phantom/camera/camera_extrinsics_xyzrpy_example.json and
phantom/camera/camera_intrinsics_example.json for filled-in examples
generated from the repo's existing calibration.

Usage:
    python test_ur7e_single_pose_overlay.py --image path/to/photo.jpg \
        --pos 0.5 0.0 0.35 [--ori 0 1 0 0] [--gripper 0.04] \
        [--camera-extrinsics path/to/extrinsics_xyzrpy.json] \
        [--camera-intrinsics path/to/intrinsics.json] [options]
"""
import argparse
import json
import logging
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.twin_robot import TwinRobot, MujocoCameraParams, convert_real_camera_ori_to_mujoco

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKING_ERROR_THRESHOLD = 0.05  # meters, matches robotinpaint_processor.py

BASE_ORI_XYZW = np.array([0, 1, 0, 0])  # gripper facing down

DEFAULT_CAMERA_EXTRINSICS_PATH = os.path.join(
    REPO_ROOT, "phantom/camera/camera_extrinsics_xyzrpy_example.json"
)
DEFAULT_CAMERA_INTRINSICS_PATH = os.path.join(
    REPO_ROOT, "phantom/camera/camera_intrinsics_example.json"
)


def load_camera_extrinsics_xyzrpy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera extrinsics from a {"xyz": [...], "rpy": [...], "degrees": bool}
    JSON file and return (position, 3x3 rotation matrix in real-world coords).
    """
    with open(path) as f:
        data = json.load(f)
    pos = np.array(data["xyz"], dtype=float)
    rpy = np.array(data["rpy"], dtype=float)
    degrees = data.get("degrees", True)
    ori_matrix = Rotation.from_euler("xyz", rpy, degrees=degrees).as_matrix()
    return pos, ori_matrix


def load_camera_intrinsics(path: str) -> tuple[np.ndarray, tuple]:
    """
    Load camera intrinsics from a {"camera_matrix": [...], "dist_coefs": [...],
    "resolution": [w, h]} JSON file and return (3x3 camera matrix, (w, h)
    the calibration was done at, or None if "resolution" is absent).
    """
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=float)
    resolution = tuple(data["resolution"]) if "resolution" in data else None
    return camera_matrix, resolution


def build_camera_params(camera_extrinsics_path: str, camera_intrinsics_path: str,
                         img_w: int, img_h: int) -> MujocoCameraParams:
    """
    Builds MujocoCameraParams for a photo of size (img_w, img_h), scaling
    the intrinsics file's camera matrix from its own calibration resolution
    to (img_w, img_h) if given (mirrors test_pushT_twin_overlay.py's
    build_camera_params).
    """
    camera_pos, camera_ori_matrix = load_camera_extrinsics_xyzrpy(camera_extrinsics_path)
    camera_matrix, calib_resolution = load_camera_intrinsics(camera_intrinsics_path)

    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    if calib_resolution is not None:
        calib_w, calib_h = calib_resolution
        sx, sy = img_w / calib_w, img_h / calib_h
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    sensor_width, sensor_height = img_w / fy / 1000, img_h / fx / 1000

    # MuJoCo's pinhole camera has no lens-distortion model, so dist_coefs is
    # ignored -- the composited robot mask will be geometrically approximate
    # near the frame edges where distortion is largest.
    v_fov = np.degrees(2 * np.arctan(img_h / (2 * fy)))

    camera_ori_wxyz = convert_real_camera_ori_to_mujoco(camera_ori_matrix)

    return MujocoCameraParams(
        name="frontview",
        pos=camera_pos,
        ori_wxyz=camera_ori_wxyz,
        fov=v_fov,
        resolution=(img_h, img_w),
        sensorsize=np.array([sensor_width, sensor_height]),
        principalpixel=np.array([img_w / 2 - cx, cy - img_h / 2]),
        focalpixel=np.array([fx, fy]),
    )


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
    parser.add_argument("--image", required=True, help="Path to input photo (should contain no hands/arms)")
    parser.add_argument("--output", default=None, help="Output image path (default: <image_stem>_ur7e_pose.png)")
    parser.add_argument("--robot", default="UR7e")
    parser.add_argument("--gripper", default="Robotiq85")
    parser.add_argument("--pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                         help="Target end-effector position relative to robot base (meters)")
    parser.add_argument("--ori", type=float, nargs=4, default=list(BASE_ORI_XYZW), metavar=("X", "Y", "Z", "W"),
                         help="Target end-effector orientation quaternion xyzw (default: gripper facing down)")
    parser.add_argument("--gripper-pos", type=float, default=0.04,
                         help="Gripper opening distance in meters (0=closed, 0.085=fully open)")
    parser.add_argument("--camera-extrinsics", default=DEFAULT_CAMERA_EXTRINSICS_PATH,
                         help="Path to a JSON file with {\"xyz\": [...], \"rpy\": [...], \"degrees\": bool} "
                              "camera extrinsics")
    parser.add_argument("--camera-intrinsics", default=DEFAULT_CAMERA_INTRINSICS_PATH,
                         help="Path to a JSON file with {\"camera_matrix\": [...], \"dist_coefs\": [...], "
                              "\"resolution\": [w, h]} camera intrinsics")
    parser.add_argument("--render", action="store_true", help="Open an on-screen MuJoCo viewer while stepping")
    args = parser.parse_args()

    output_path = args.output or (os.path.splitext(args.image)[0] + "_ur7e_pose.png")

    print(f"Reading image: {args.image}")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_h, img_w = image_bgr.shape[:2]
    frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    camera_params = build_camera_params(args.camera_extrinsics, args.camera_intrinsics, img_w, img_h)
    print(f"Camera params: pos={camera_params.pos}, resolution={camera_params.resolution}")

    pos = np.array(args.pos)
    ori_xyzw = np.array(args.ori)
    print(f"Target pose: pos={pos}, ori_xyzw={ori_xyzw}, gripper_pos={args.gripper_pos}")

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
        square=False,
    )
    print("TwinRobot initialized successfully.\n")

    try:
        state = {"pos": pos, "ori_xyzw": ori_xyzw, "gripper_pos": args.gripper_pos}
        robot_results = robot.move_to_target_state(state, init=True)

        if robot_results["pos_err"] > TRACKING_ERROR_THRESHOLD:
            logger.warning(f"Tracking error too large ({robot_results['pos_err']:.4f} m) -- "
                            "target pose may be unreachable, overlay geometry may be off")

        output_frame = overlay_robot(frame, robot_results)
    finally:
        robot.close()

    print(f"Writing output to {output_path}")
    cv2.imwrite(output_path, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
    print(f"\nDone. Position tracking error: {robot_results['pos_err']:.4f} m")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
