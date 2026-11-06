"""Start(random) -> approach -> grasp -> lift -> transport -> place -> release -> retreat trajectory generation."""
import numpy as np
from scipy.spatial.transform import Rotation

BASE_ORI_XYZW = np.array([0, 1, 0, 0])  # gripper facing down
# Input domain TwinRobot._convert_handgripper_pos_to_action expects, for
# either supported gripper on this rig: Robotiq85 (parallel jaw, where this
# really is a width in meters) or ECBPiUR (Schmalz vacuum gripper -- the one
# actually mounted on the UR7e per Cad_files/gripper_traceparts/10_03_01_00504.txt,
# where it's just an idle/engage command riding the same [0, 0.085] range).
GRIPPER_IDLE = 0.085    # released / suction off
GRIPPER_ENGAGED = 0.0   # holding the object / suction on


# LIFT/TRANSPORT/PLACE keep the ids they had before approach, grasp, release and
# retreat existed, so datasets and consumers written against them stay valid.
STAGE_LIFT, STAGE_TRANSPORT, STAGE_PLACE = 0, 1, 2
STAGE_APPROACH, STAGE_GRASP, STAGE_RELEASE, STAGE_RETREAT = 3, 4, 5, 6


def generate_pickplace_trajectory(region_low: np.ndarray, region_high: np.ndarray, place_pos: np.ndarray,
                                   lift_height: float, velocity_mps: float, fps: float, release_frames: int,
                                   yaw_range_deg: float, rng: np.random.Generator,
                                   yaw_center_deg: float = 0.0, grasp_frames: int = 8,
                                   approach_height: float = None, settle_frames: int = 3) -> dict:
    """
    Build a start(random) -> approach -> descend -> grasp -> lift -> transport -> place
    -> release -> retreat trajectory.

    The robot starts at a random pose in [region_low, region_high]. The object sits at
    a separate random xy on the picking plane (z = region_low[2]); the robot flies
    above it at `approach_height` (default `lift_height`), descends, dwells for
    `grasp_frames` with suction on, then carries it to `place_pos` and releases it
    there for `release_frames` before retreating straight up.

    Suction only switches after the arm has been held still for `settle_frames` at the
    pick (or place) pose: the controller trails a moving target, so on the first dwell
    frame the cup is still above the object, and engaging there would lock that gap into
    the grasp and then press the object into the table as the arm catches up (likewise
    releasing early drops it from above the table). The settle frames at the pick count
    as GRASP, the ones at the place as PLACE.

    Returns dict with per-frame `positions` (N,3), `quats_xyzw` (N,4),
    `gripper_cmds` (N,) (in the [GRIPPER_ENGAGED, GRIPPER_IDLE] domain
    TwinRobot._convert_handgripper_pos_to_action expects), `suction` (N,) bool
    (gripper_cmds == GRIPPER_ENGAGED), and `stage` (N,) int in {STAGE_*}.
    Note PLACE now ends when the object arrives; the release dwell that used to
    count as PLACE is STAGE_RELEASE.

    Also returns `start_pos`, `pick_pos` and `object_quat_xyzw` (the object is
    yawed with the gripper so the cup sits squarely on it).
    """
    region_low, region_high = np.asarray(region_low, float), np.asarray(region_high, float)
    approach_height = lift_height if approach_height is None else float(approach_height)

    start_pos = rng.uniform(region_low, region_high)
    pick_xy = rng.uniform(region_low[:2], region_high[:2])
    pick_pos = np.array([pick_xy[0], pick_xy[1], region_low[2]])
    # yaw_center_deg shifts the whole randomization band. It matters for the wrist
    # camera: the camera is bolted to the flange, so yaw rotates its footprint, and a
    # band centred away from the orientation the mosaic was scanned at keeps the view
    # turned away from the scanned patch (phantom/synthetic/wrist_region.py).
    yaw = yaw_center_deg + rng.uniform(-yaw_range_deg, yaw_range_deg)
    ori_xyzw = (Rotation.from_quat(BASE_ORI_XYZW) * Rotation.from_euler("z", yaw, degrees=True)).as_quat()

    above_pick = np.array([pick_pos[0], pick_pos[1], approach_height])
    above_place = np.array([place_pos[0], place_pos[1], lift_height])
    place_target = np.asarray(place_pos, dtype=float)

    def segment(p_from, p_to):
        dist = np.linalg.norm(p_to - p_from)
        n = max(2, int(round(dist / velocity_mps * fps)))
        t = np.linspace(0, 1, n, endpoint=False)
        return p_from[None, :] + t[:, None] * (p_to - p_from)[None, :]

    # (positions, gripper command, stage) per phase, in time order.
    phases = [
        (segment(start_pos, above_pick), GRIPPER_IDLE, STAGE_APPROACH),
        (segment(above_pick, pick_pos), GRIPPER_IDLE, STAGE_APPROACH),
        (np.repeat(pick_pos[None, :], settle_frames, axis=0), GRIPPER_IDLE, STAGE_GRASP),
        (np.repeat(pick_pos[None, :], grasp_frames, axis=0), GRIPPER_ENGAGED, STAGE_GRASP),
        (segment(pick_pos, above_pick), GRIPPER_ENGAGED, STAGE_LIFT),
        (segment(above_pick, above_place), GRIPPER_ENGAGED, STAGE_TRANSPORT),
        (segment(above_place, place_target), GRIPPER_ENGAGED, STAGE_PLACE),
        (np.repeat(place_target[None, :], settle_frames, axis=0), GRIPPER_ENGAGED, STAGE_PLACE),
        (np.repeat(place_target[None, :], release_frames, axis=0), GRIPPER_IDLE, STAGE_RELEASE),
        (segment(place_target, above_place), GRIPPER_IDLE, STAGE_RETREAT),
    ]
    positions = np.concatenate([p for p, _, _ in phases], axis=0)
    gripper_cmds = np.concatenate([np.full(len(p), g) for p, g, _ in phases])
    stage = np.concatenate([np.full(len(p), s) for p, _, s in phases])
    quats_xyzw = np.repeat(ori_xyzw[None, :], len(positions), axis=0)

    return {
        "positions": positions,
        "quats_xyzw": quats_xyzw,
        "gripper_cmds": gripper_cmds,
        "suction": gripper_cmds == GRIPPER_ENGAGED,
        "stage": stage,
        "start_pos": start_pos,
        "pick_pos": pick_pos,
        "object_quat_xyzw": ori_xyzw,
    }
