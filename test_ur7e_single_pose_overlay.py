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

Optionally, a second wrist-mounted camera can also be tested by passing
--wrist-image (a second photo of the same scene/pose, taken from a camera
mounted on the robot's flange) along with --wrist-camera-extrinsics /
--wrist-camera-intrinsics. The wrist camera's extrinsics use the same
{xyz, rpy, degrees} JSON format as the fixed camera, but are interpreted
relative to the flange (eef) body frame instead of world coordinates --
the camera is attached to the flange in MuJoCo, so it moves with the arm.

A real wrist photo only exists for the one pose it was shot at, so for an
arbitrary target pose pass --wrist-mosaic <model_dir> instead of
--wrist-image: the wrist background is then *synthesized at the pose the
robot actually reached*, by projecting a scene mosaic (a stitched top-down
photo of the flat workspace, built by phantom.scene_mosaic.build_mosaic)
into the wrist camera. In that mode the wrist camera's intrinsics and its
flange-relative extrinsic come from the mosaic's own mosaic_meta.json
rather than from --wrist-camera-intrinsics / --wrist-camera-extrinsics, so
the camera MuJoCo renders the robot from and the camera the background is
rendered for are the same camera (see phantom/synthetic/wrist_background.py).
Because the twin renders all of its cameras at one resolution, the wrist
view is rendered at the fixed photo's resolution, which must therefore have
the same aspect ratio as the mosaic's calibration.

Usage:
    python test_ur7e_single_pose_overlay.py --image path/to/photo.jpg \
        --pos 0.5 0.0 0.35 [--ori 0 1 0 0] [--gripper 0.04] \
        [--camera-extrinsics path/to/extrinsics_xyzrpy.json] \
        [--camera-intrinsics path/to/intrinsics.json] \
        [--wrist-image path/to/wrist_photo.jpg] \
        [--wrist-camera-extrinsics path/to/wrist_extrinsics_xyzrpy.json] \
        [--wrist-camera-intrinsics path/to/wrist_intrinsics.json] [options]

    python test_ur7e_single_pose_overlay.py --image path/to/photo.jpg \
        --pos 0.5 0.0 0.35 --wrist-mosaic phantom/scene_mosaic/out_ep1_full
"""
import argparse
import json
import logging
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.synthetic.wrist_background import WristMosaicBackground
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
DEFAULT_WRIST_CAMERA_EXTRINSICS_PATH = os.path.join(
    REPO_ROOT, "phantom/camera/camera_extrinsics_xyzrpy_wrist_example.json"
)
DEFAULT_WRIST_CAMERA_INTRINSICS_PATH = os.path.join(
    REPO_ROOT, "phantom/camera/camera_intrinsics_wrist_example.json"
)
DEFAULT_WRIST_MOSAIC_MODEL = os.path.join(REPO_ROOT, "phantom/scene_mosaic/out_ep1_full")


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
                         img_w: int, img_h: int, name: str = "frontview") -> MujocoCameraParams:
    """
    Builds MujocoCameraParams for a photo of size (img_w, img_h), scaling
    the intrinsics file's camera matrix from its own calibration resolution
    to (img_w, img_h) if given (mirrors test_pushT_twin_overlay.py's
    build_camera_params).

    `name` is the MuJoCo camera name -- "frontview" for the fixed world
    camera, "wristview" for the flange-mounted one. The extrinsics/pos are
    interpreted the same way regardless of name; it's the caller's
    responsibility to know whether they're in world or flange-local
    coordinates (see TwinRobot's `wrist_camera_params` for the latter).
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
        name=name,
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
    parser.add_argument("--wrist-image", default=None,
                         help="Optional path to a second photo taken from a wrist-mounted camera at the same "
                              "robot pose. If given, the robot is also rendered from a camera attached to the "
                              "flange and composited onto this image.")
    parser.add_argument("--wrist-output", default=None,
                         help="Output image path for the wrist-camera overlay (default: <wrist_image_stem>_ur7e_pose.png)")
    parser.add_argument("--wrist-camera-extrinsics", default=DEFAULT_WRIST_CAMERA_EXTRINSICS_PATH,
                         help="Path to a JSON file with {\"xyz\": [...], \"rpy\": [...], \"degrees\": bool} "
                              "wrist camera extrinsics, expressed relative to the flange (eef) frame, not world")
    parser.add_argument("--wrist-camera-intrinsics", default=DEFAULT_WRIST_CAMERA_INTRINSICS_PATH,
                         help="Path to a JSON file with {\"camera_matrix\": [...], \"dist_coefs\": [...], "
                              "\"resolution\": [w, h]} wrist camera intrinsics")
    parser.add_argument("--wrist-mosaic", nargs="?", const=DEFAULT_WRIST_MOSAIC_MODEL, default=None,
                         metavar="MODEL_DIR",
                         help="Render the wrist-camera background from a scene mosaic model directory "
                              "(mosaic.png, mosaic_valid.png, mosaic_meta.json, gain.npy) at the pose the "
                              "robot reaches, instead of compositing onto a fixed --wrist-image. Passed "
                              f"without a value, uses {os.path.relpath(DEFAULT_WRIST_MOSAIC_MODEL, REPO_ROOT)}. "
                              "The wrist camera's intrinsics/extrinsic then come from the mosaic, not from "
                              "--wrist-camera-intrinsics/--wrist-camera-extrinsics.")
    parser.add_argument("--wrist-extrinsics-source", choices=("mosaic", "json"), default="mosaic",
                         help="With --wrist-mosaic: which flange -> camera transform to mount the wrist camera "
                              "with. \"mosaic\" (default) uses mosaic_meta.json's self-calibrated T_eef_cam; "
                              "\"json\" uses --wrist-camera-extrinsics instead, so the hand-eye calibration can "
                              "be checked against the scan. Either way the background is rendered for the pose "
                              "the twin reports, so robot and background always share one camera -- what changes "
                              "is where on the mosaic that camera is deemed to be looking.")
    parser.add_argument("--wrist-apply-distortion", action=argparse.BooleanOptionalAction, default=False,
                         help="With --wrist-mosaic: re-apply the real lens distortion to the background. "
                              "Default (off) renders a pinhole background, matching MuJoCo's camera model, "
                              "so the composited robot stays geometrically consistent with its background.")
    parser.add_argument("--wrist-apply-gain", action=argparse.BooleanOptionalAction, default=True,
                         help="With --wrist-mosaic: re-apply the camera-fixed shading (vignetting, the arm's "
                              "own shadow) that was divided out when the mosaic was built")
    parser.add_argument("--wrist-fill-color", type=int, nargs=3, default=[0, 0, 0], metavar=("R", "G", "B"),
                         help="With --wrist-mosaic: colour for pixels the scan never saw")
    parser.add_argument("--wrist-mount-translation", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                         help="With --wrist-mosaic: override the camera's lever arm in the flange frame (m); "
                              "default uses mosaic_meta.json's T_eef_cam translation (the hand-eye value)")
    parser.add_argument("--wrist-background-output", default=None,
                         help="With --wrist-mosaic: also write the raw rendered background (no robot) here")
    parser.add_argument("--show-wrist-camera", action="store_true",
                         help="Render a box at the wrist camera's pose so it is visible in the fixed-camera overlay "
                              "(requires --wrist-image)")
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

    if args.wrist_mosaic and args.wrist_image:
        parser.error("--wrist-mosaic and --wrist-image are mutually exclusive: the background is either "
                     "synthesized from the mosaic at the reached pose, or taken from the given photo")

    wrist_frame = None
    wrist_camera_params = None
    wrist_bg = None
    if args.wrist_mosaic:
        # Configure the wrist camera from the same mosaic its background is rendered
        # from, so the two cannot disagree. The twin renders every camera at one
        # resolution (TwinRobot passes a single camera_heights/widths), hence the fixed
        # photo's size is also the wrist render size.
        print(f"Loading wrist mosaic model: {args.wrist_mosaic}")
        T_eef_cam = None
        if args.wrist_extrinsics_source == "json":
            # Only the extrinsic is taken from the file: the intrinsics stay the
            # mosaic's, because the background is reprojected with them and the twin's
            # camera has to match it pixel for pixel.
            cam_pos, cam_rot = load_camera_extrinsics_xyzrpy(args.wrist_camera_extrinsics)
            T_eef_cam = np.eye(4)
            T_eef_cam[:3, :3], T_eef_cam[:3, 3] = cam_rot, cam_pos
            print(f"Using wrist extrinsics from {args.wrist_camera_extrinsics} instead of the mosaic's: "
                  f"xyz={cam_pos}")
        wrist_bg = WristMosaicBackground(
            args.wrist_mosaic, img_w, img_h, square=False,
            apply_distortion=args.wrist_apply_distortion,
            apply_gain=args.wrist_apply_gain,
            fill_color=args.wrist_fill_color,
            mount_translation=args.wrist_mount_translation,
            T_eef_cam=T_eef_cam,
        )
        print(wrist_bg.describe())
        wrist_camera_params = wrist_bg.camera_params(name="wristview")
        print(f"Wrist camera params (flange-relative, from mosaic): pos={wrist_camera_params.pos}, "
              f"resolution={wrist_camera_params.resolution}")
    elif args.wrist_image:
        print(f"Reading wrist image: {args.wrist_image}")
        wrist_image_bgr = cv2.imread(args.wrist_image)
        if wrist_image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {args.wrist_image}")
        wrist_img_h, wrist_img_w = wrist_image_bgr.shape[:2]
        wrist_frame = cv2.cvtColor(wrist_image_bgr, cv2.COLOR_BGR2RGB)

        if (wrist_img_w, wrist_img_h) != (img_w, img_h):
            print(f"WARNING: wrist photo is {wrist_img_w}x{wrist_img_h} but the twin renders every camera at "
                  f"the fixed photo's {img_w}x{img_h} -- the wrist render will be resized to composite.")

        wrist_camera_params = build_camera_params(args.wrist_camera_extrinsics, args.wrist_camera_intrinsics,
                                                    wrist_img_w, wrist_img_h, name="wristview")
        print(f"Wrist camera params (flange-relative): pos={wrist_camera_params.pos}, "
              f"resolution={wrist_camera_params.resolution}")

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
        wrist_camera_params=wrist_camera_params,
        show_wrist_camera=args.show_wrist_camera and wrist_camera_params is not None,
    )
    print("TwinRobot initialized successfully.\n")

    try:
        state = {"pos": pos, "ori_xyzw": ori_xyzw, "gripper_pos": args.gripper_pos}
        robot_results = robot.move_to_target_state(state, init=True)

        if robot_results["pos_err"] > TRACKING_ERROR_THRESHOLD:
            logger.warning(f"Tracking error too large ({robot_results['pos_err']:.4f} m) -- "
                            "target pose may be unreachable, overlay geometry may be off")

        output_frame = overlay_robot(frame, robot_results)

        # Taking the render pose out of the simulation (rather than recomposing it from
        # the commanded pose) guarantees the background is rendered for exactly the
        # camera MuJoCo rendered the robot from -- see TwinRobot.get_wrist_camera_pose.
        wrist_coverage = None
        if wrist_bg is not None:
            wrist_frame, wrist_coverage = wrist_bg.render_at_camera(robot.get_wrist_camera_pose())

        wrist_output_frame = None
        if wrist_frame is not None:
            wrist_output_frame = overlay_robot(wrist_frame, {
                "rgb_img": robot_results["wrist_rgb_img"],
                "robot_mask": robot_results["wrist_robot_mask"],
                "gripper_mask": robot_results["wrist_gripper_mask"],
            })
    finally:
        robot.close()

    print(f"Writing output to {output_path}")
    cv2.imwrite(output_path, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
    print(f"\nDone. Position tracking error: {robot_results['pos_err']:.4f} m")
    print(f"Output written to: {output_path}")

    if wrist_output_frame is not None:
        default_wrist_stem = os.path.splitext(args.wrist_image)[0] if args.wrist_image \
            else os.path.splitext(args.image)[0] + "_wrist"
        wrist_output_path = args.wrist_output or (default_wrist_stem + "_ur7e_pose.png")
        print(f"Writing wrist-camera output to {wrist_output_path}")
        cv2.imwrite(wrist_output_path, cv2.cvtColor(wrist_output_frame, cv2.COLOR_RGB2BGR))
        if wrist_coverage is not None:
            print(f"Wrist mosaic coverage: {wrist_coverage * 100:.1f}% of the view "
                  "(the rest is workspace the scan never saw)")
        if args.wrist_background_output and wrist_frame is not None:
            cv2.imwrite(args.wrist_background_output, cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR))
            print(f"Wrist-camera background (no robot) written to: {args.wrist_background_output}")
        print(f"Wrist-camera output written to: {wrist_output_path}")


if __name__ == "__main__":
    main()
