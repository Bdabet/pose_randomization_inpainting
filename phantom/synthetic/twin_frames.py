"""
The frame conventions that relate a recorded robot pose to the twin's commands and to
the wrist camera.

Three frames are in play and no two of them coincide:

- the **flange** (robosuite's eef *body*): what a real robot's `robot_eef_pose`
  reports in this stack, and what `T_eef_cam` is relative to -- the camera is bolted
  to it, so it is the frame everything about the wrist view is expressed in;
- the **grip site** (`robot0_eef_pos/quat`): what `TwinRobot.move_to_target_state`
  tracks, one tool length further along the tool axis (0.155 m for a Robotiq85);
- the **commanded orientation**: not the orientation the arm settles at. The
  controller right-multiplies a fixed twist about the tool axis (45 deg for a
  Robotiq85, from `TwinRobot.get_action_from_ee_pose`'s -135 deg about z).

Both constants change with `gripper`, so they are measured from the running simulation
rather than written down here. Getting the second one wrong is quiet and expensive: the
arm still goes where you asked, but anything computed from the orientation you *think*
it has -- above all where the wrist camera points -- is rotated away from reality.

Used by the synthetic processor (`phantom/processors/synthetic_processor.py`), the
coverage preflight (`phantom/synthetic/wrist_region.py`) and the recorded-trajectory
replay (`scripts/replay_real_trajectory.py`), so all three agree by construction.
"""
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Gripper facing down, the orientation trajectory.py builds its yaw band around.
BASE_ORI_XYZW = np.array([0.0, 1.0, 0.0, 0.0])


def measure_tool_frames(robot) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure (T_flange_grip, ori_correction) from a `TwinRobot`.

    `T_flange_grip` is the grip site in the flange frame; `ori_correction` is the
    rotation from a commanded orientation to the one the arm settles at, so that
    settled = commanded @ ori_correction.

    One settle at the robot's current position is enough for both, and it leaves the
    arm somewhere reachable.
    """
    obs = robot.env.env._get_observations()
    probe_ori = BASE_ORI_XYZW
    robot.move_to_target_state({"pos": obs["robot0_eef_pos"] - robot.robot_base_pos,
                                "ori_xyzw": probe_ori, "gripper_pos": 0.04}, init=True)
    obs = robot.env.env._get_observations()

    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
    T_base_grip[:3, 3] = obs["robot0_eef_pos"] - robot.robot_base_pos
    T_flange_grip = np.linalg.inv(robot.get_flange_pose()) @ T_base_grip
    ori_correction = (Rotation.from_quat(probe_ori).inv()
                      * Rotation.from_quat(obs["robot0_eef_quat"])).as_matrix()
    return T_flange_grip, ori_correction


def describe_tool_frames(T_flange_grip: np.ndarray, ori_correction: np.ndarray) -> str:
    """One-line summary for a run's log."""
    return (f"tool length {np.linalg.norm(T_flange_grip[:3, 3]):.3f} m, "
            f"commanded-vs-settled twist "
            f"{np.degrees(Rotation.from_matrix(ori_correction).magnitude()):.1f} deg")


def flange_to_command(pose6: Sequence[float], T_flange_grip: np.ndarray,
                      ori_correction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recorded flange pose ([x, y, z, rx, ry, rz], base frame) -> the (pos, ori_xyzw)
    `TwinRobot.move_to_target_state` takes, such that the arm settles with its flange
    on the recorded pose.
    """
    from phantom.scene_mosaic import pose6_to_matrix  # local: pulls in cv2/zarr

    T_base_grip = pose6_to_matrix(np.asarray(pose6, dtype=float)) @ T_flange_grip
    ori = Rotation.from_matrix(T_base_grip[:3, :3] @ np.linalg.inv(ori_correction)).as_quat()
    return T_base_grip[:3, 3], ori


def command_to_flange(pos: Sequence[float], ori_xyzw: Sequence[float], T_flange_grip: np.ndarray,
                      ori_correction: np.ndarray) -> np.ndarray:
    """
    The inverse of `flange_to_command`: a (pos, ori_xyzw) as `TwinRobot.move_to_target_state`
    takes it -> the flange pose [x, y, z, rx, ry, rz] (base frame, rotation vector) the arm
    settles at for it. This is what a real robot running with its TCP at the flange would
    have been commanded, so synthetic actions written this way match recorded ones.
    """
    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = Rotation.from_quat(np.asarray(ori_xyzw, dtype=float)).as_matrix() @ ori_correction
    T_base_grip[:3, 3] = np.asarray(pos, dtype=float)
    T_base_flange = T_base_grip @ np.linalg.inv(T_flange_grip)
    return np.concatenate([T_base_flange[:3, 3], Rotation.from_matrix(T_base_flange[:3, :3]).as_rotvec()])


def grip_to_camera(T_flange_grip: np.ndarray, T_eef_cam: np.ndarray,
                   pos: Sequence[float], ori_xyzw: Sequence[float],
                   ori_correction: np.ndarray = None) -> np.ndarray:
    """
    T_base_cam for a trajectory waypoint, as `TwinRobot.move_to_target_state` takes it.

    This is the forward direction of `flange_to_command`: it answers "if I command
    this, where does the wrist camera end up", which is what coverage prediction needs.
    Pass the measured `ori_correction`, or the answer is for a camera pointing
    somewhere the robot never goes.
    """
    R_cmd = Rotation.from_quat(np.asarray(ori_xyzw, dtype=float)).as_matrix()
    T_base_grip = np.eye(4)
    T_base_grip[:3, :3] = R_cmd if ori_correction is None else R_cmd @ ori_correction
    T_base_grip[:3, 3] = np.asarray(pos, dtype=float)
    return T_base_grip @ np.linalg.inv(T_flange_grip) @ T_eef_cam
