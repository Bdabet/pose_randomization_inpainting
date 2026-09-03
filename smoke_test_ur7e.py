"""
One-off validation script for the new UR7e twin robot (Phase 5, step 2 of the
UR7e integration plan). Not part of the pipeline -- run manually after
activating the `phantom` conda env (see install.sh) to sanity-check that:

  1. The UR7e MJCF loads inside robosuite/MuJoCo without errors.
  2. TwinRobot can drive it to a handful of end-effector poses across the
     workspace with low tracking error (checks kinematics/actuation).
  3. The instance-segmentation IDs used by TwinRobot.get_robot_mask/
     get_gripper_mask (1 = robot, 3 = gripper) still hold for UR7e's scene.

Usage:
    python smoke_test_ur7e.py [--render]

Outputs PNGs (RGB render + colorized segmentation mask) per test pose into
./smoke_test_ur7e_out/ for visual inspection.
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation

from phantom.twin_robot import TwinRobot, MujocoCameraParams

OUT_DIR = "smoke_test_ur7e_out"

# Arbitrary but sane frontview camera looking at the robot's workspace.
# This does NOT need to match the real calibrated camera used by the actual
# pipeline (see robotinpaint_processor.py's _get_mujoco_camera_params for
# that) -- it only needs to see the robot for this geometry/kinematics check.
CAMERA_PARAMS = MujocoCameraParams(
    name="frontview",
    pos=np.array([1.6, 0.0, 1.4]),
    ori_wxyz=Rotation.from_euler("xyz", [0, 35, 90], degrees=True).as_quat(scalar_first=True),
    fov=60.0,
    resolution=(480, 640),
    sensorsize=np.array([36.0, 24.0]),
    principalpixel=np.array([0.0, 0.0]),
    focalpixel=np.array([600.0, 600.0]),
)

# A handful of end-effector poses spanning the workspace: center, left, right,
# near, far, plus one with a non-identity orientation.
TEST_POSES = [
    {"name": "center", "pos": np.array([0.5, 0.0, 0.3]), "quat_xyzw": np.array([0, 1, 0, 0]), "gripper": 0.04},
    {"name": "left", "pos": np.array([0.5, 0.25, 0.3]), "quat_xyzw": np.array([0, 1, 0, 0]), "gripper": 0.0},
    {"name": "right", "pos": np.array([0.5, -0.25, 0.3]), "quat_xyzw": np.array([0, 1, 0, 0]), "gripper": 0.08},
    {"name": "near", "pos": np.array([0.35, 0.0, 0.25]), "quat_xyzw": np.array([0, 1, 0, 0]), "gripper": 0.04},
    {"name": "far", "pos": np.array([0.65, 0.0, 0.4]), "quat_xyzw": np.array([0, 1, 0, 0]), "gripper": 0.04},
    {
        "name": "rotated",
        "pos": np.array([0.5, 0.0, 0.35]),
        "quat_xyzw": (Rotation.from_quat([0, 1, 0, 0]) * Rotation.from_euler("z", 45, degrees=True)).as_quat(),
        "gripper": 0.04,
    },
]

TRACKING_ERROR_THRESHOLD = 0.05  # meters, matches robotinpaint_processor.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open an on-screen MuJoCo viewer while stepping")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Instantiating TwinRobot(robot_name='UR7e', gripper_name='Robotiq85')...")
    robot = TwinRobot(
        robot_name="UR7e",
        gripper_name="Robotiq85",
        camera_params=CAMERA_PARAMS,
        camera_height=CAMERA_PARAMS.resolution[0],
        camera_width=CAMERA_PARAMS.resolution[1],
        render=args.render,
        n_steps_short=3,
        n_steps_long=75,
    )
    print("TwinRobot initialized successfully.\n")

    import cv2

    all_ok = True
    for i, pose in enumerate(TEST_POSES):
        state = {"pos": pose["pos"], "ori_xyzw": pose["quat_xyzw"], "gripper_pos": pose["gripper"]}
        result = robot.move_to_target_state(state, init=(i == 0))

        pos_err = result["pos_err"]
        status = "OK" if pos_err <= TRACKING_ERROR_THRESHOLD else "FAIL (tracking error too high)"
        if pos_err > TRACKING_ERROR_THRESHOLD:
            all_ok = False
        print(f"[{pose['name']}] pos_err={pos_err:.4f} m -> {status}")

        rgb = cv2.cvtColor(result["rgb_img"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, f"{i:02d}_{pose['name']}_rgb.png"), rgb)

        robot_px = int(result["robot_mask"].sum())
        gripper_px = int(result["gripper_mask"].sum())
        print(f"  robot_mask (id=1) pixels: {robot_px}, gripper_mask (id=3) pixels: {gripper_px}")
        if robot_px == 0 or gripper_px == 0:
            all_ok = False
            print("  WARNING: expected instance ID (1=robot, 3=gripper) produced an empty mask --"
                  " the segmentation IDs may have shifted for UR7e's scene composition.")

        cv2.imwrite(os.path.join(OUT_DIR, f"{i:02d}_{pose['name']}_robot_mask.png"), result["robot_mask"] * 255)
        cv2.imwrite(os.path.join(OUT_DIR, f"{i:02d}_{pose['name']}_gripper_mask.png"), result["gripper_mask"] * 255)

    robot.close()

    print(f"\nRendered frames written to ./{OUT_DIR}/ -- inspect visually for pose/orientation correctness,")
    print("no clipping through table/mount, and that instance IDs 1 (robot) / 3 (gripper) are present as expected.")
    if not all_ok:
        print("\nWARNING: one or more poses exceeded the tracking-error threshold -- check init_qpos/kinematics.")


if __name__ == "__main__":
    main()
