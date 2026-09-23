"""
Generate synthetic pick-and-place demonstrations for domain/pose randomization.

For each episode:
  1. Sample a random start end-effector pose (position + yaw) inside a
     user-defined region -- this is the "not seen in the real training data"
     part, standing in for wherever the object happened to be grasped.
  2. Lift straight up to a fixed height.
  3. Transport horizontally to a fixed place position.
  4. Descend and release the gripper.

The resulting trajectory is rendered with TwinRobot (same MuJoCo twin used by
the real pipeline's robot_inpaint stage) and composited onto a real
background video of the (robot-free) workspace -- e.g. the output of
`process_data.py mode=hand_inpaint`, or any other clean plate of the scene.
Because the workspace is assumed mostly stationary, each synthetic frame is
paired with a background frame chosen independently of trajectory time (a
held random-walk sampler, see `sample_background_indices`), so a trajectory
can be longer, shorter, or differently paced than the background clip.

Output is a single Zarr store per demo_name, matching the field layout of a
real diffusion_policy real-robot ReplayBuffer collected on this same UR
hardware (data/{action,robot_eef_pose,robot_joint,robot_joint_vel,stage,
timestamp} + meta/episode_ends -- verified against a reference
replay_buffer.zarr; no 'img' array in that layout either, since images are
kept in separate video files, not embedded in the zarr):

  <output_dir>/<demo_name>.zarr/
    data/
      action          (N, 6) float64  -- commanded [pos(3), rotvec(3)]
      robot_eef_pose  (N, 6) float64  -- tracked   [pos(3), rotvec(3)]
      robot_joint     (N, 6) float64  -- tracked joint angles (UR7e has 6)
      robot_joint_vel (N, 6) float64  -- tracked joint velocities
      gripper_cmd     (N,)   float64  -- commanded gripper value, in the
                                         [GRIPPER_ENGAGED, GRIPPER_IDLE] domain
                                         TwinRobot expects (extra vs. the
                                         reference layout -- this task has a
                                         gripper, that pushT one didn't; the
                                         actual tool on this rig is the
                                         Schmalz ECBPiUR vacuum gripper, see
                                         --gripper)
      stage           (N,)   int64    -- 0=lift, 1=transport, 2=place
      timestamp       (N,)   float64  -- seconds elapsed since episode start
    meta/
      episode_ends    (num_episodes,) int64  -- cumulative frame count per episode

Deliberately NOT included: robot_eef_pose_vel / raw img arrays. The sim's
eef velocity observables (robot0_eef_vel_lin/ang) are disabled by default in
this robosuite fork (see single_arm.py's `actives` list) and enabling them
requires poking at env internals well beyond what a rendering script should
do -- rather than fabricate it, it's left out. Joint *position* is also
inactive by default, but is reconstructed exactly via atan2(joint_pos_sin,
joint_pos_cos), which ARE active, so robot_joint is genuine, not guessed.

This can be opened directly with
`diffusion_policy.common.replay_buffer.ReplayBuffer.copy_from_path(path)`,
or with plain `zarr.open(path)` if diffusion_policy isn't installed in this
environment -- ZarrReplayBufferWriter below only depends on `zarr` and
reimplements the handful of ReplayBuffer.add_episode semantics needed to
write that same format, so no extra dependency on the diffusion_policy repo
itself is required.

A per-episode video (<episode_idx>_<Robot>.mp4) is also written next to the
zarr store, purely for quick visual QA -- it is not read back by training
code (the reference layout has no img array either; keep images out of the
zarr and in video files, consistent with it).

Usage:
    python generate_synthetic_pickplace.py \\
        --background-video data/processed/pick_and_place/0/inpaint_processor/video_human_inpaint.mkv \\
        --num-episodes 20 \\
        --region-low 0.35 -0.25 0.25 --region-high 0.65 0.25 0.45 \\
        --place-pos 0.5 0.0 0.25 --lift-height 0.5
"""
import argparse
import os

import cv2
import mediapy as media
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

from phantom.twin_robot import TwinRobot
from test_ur7e_random_overlay import build_real_camera_params, center_crop_square, overlay_robot

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKING_ERROR_THRESHOLD = 0.05  # meters, matches robotinpaint_processor.py

BASE_ORI_XYZW = np.array([0, 1, 0, 0])  # gripper facing down
# Input domain TwinRobot._convert_handgripper_pos_to_action expects, for
# either supported gripper on this rig: Robotiq85 (parallel jaw, where this
# really is a width in meters) or ECBPiUR (Schmalz vacuum gripper -- the one
# actually mounted on the UR7e per Cad_files/gripper_traceparts/10_03_01_00504.txt,
# where it's just an idle/engage command riding the same [0, 0.085] range).
GRIPPER_IDLE = 0.085    # released / suction off
GRIPPER_ENGAGED = 0.0   # holding the object / suction on


class ZarrReplayBufferWriter:
    """
    Minimal writer for the diffusion_policy ReplayBuffer zarr layout: a
    'data' group of equal-length (N, ...) arrays plus a 'meta/episode_ends'
    array of per-episode cumulative frame counts. Episodes are appended one
    at a time; array/chunk creation is lazy, inferred from the first episode.
    """
    COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    def __init__(self, path: str):
        self.root = zarr.open(path, mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        if "episode_ends" not in self.meta:
            self.meta.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype="i8")

    def add_episode(self, episode_data: dict) -> None:
        lengths = {len(v) for v in episode_data.values()}
        assert len(lengths) == 1, f"all arrays in an episode must share length N, got {lengths}"
        n_new = lengths.pop()

        for key, value in episode_data.items():
            value = np.asarray(value)
            if key not in self.data:
                chunk_len = min(100, max(1, n_new))
                self.data.create_dataset(
                    key, shape=(0,) + value.shape[1:], chunks=(chunk_len,) + value.shape[1:],
                    dtype=value.dtype, compressor=self.COMPRESSOR,
                )
            self.data[key].append(value, axis=0)

        prev_end = self.meta["episode_ends"][-1] if len(self.meta["episode_ends"]) else 0
        self.meta["episode_ends"].append(np.array([prev_end + n_new], dtype="i8"))


def sample_background_indices(num_frames_out: int, num_bg_frames: int, hold_min: int, hold_max: int,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Independent-random background sampling, but each randomly chosen frame is
    held for a few output frames instead of re-rolled every single frame.
    Keeps the "random frame from the mostly-static video" behavior while
    avoiding single-frame flicker from sensor noise/auto-exposure.
    """
    idxs = []
    while len(idxs) < num_frames_out:
        idx = int(rng.integers(0, num_bg_frames))
        hold = int(rng.integers(hold_min, hold_max + 1))
        idxs.extend([idx] * hold)
    return np.array(idxs[:num_frames_out])


STAGE_LIFT, STAGE_TRANSPORT, STAGE_PLACE = 0, 1, 2


def generate_pickplace_trajectory(region_low: np.ndarray, region_high: np.ndarray, place_pos: np.ndarray,
                                   lift_height: float, velocity_mps: float, fps: float, release_frames: int,
                                   yaw_range_deg: float, rng: np.random.Generator) -> dict:
    """
    Build a start(random) -> lift -> transport -> place trajectory.

    Returns dict with per-frame `positions` (N,3), `quats_xyzw` (N,4),
    `gripper_cmds` (N,) (in the [GRIPPER_ENGAGED, GRIPPER_IDLE] domain
    TwinRobot._convert_handgripper_pos_to_action expects), and `stage` (N,)
    int in {STAGE_LIFT, STAGE_TRANSPORT, STAGE_PLACE} (descend + release
    both count as "place").
    """
    start_pos = rng.uniform(region_low, region_high)
    yaw = rng.uniform(-yaw_range_deg, yaw_range_deg)
    ori_xyzw = (Rotation.from_quat(BASE_ORI_XYZW) * Rotation.from_euler("z", yaw, degrees=True)).as_quat()

    lift_pos = np.array([start_pos[0], start_pos[1], lift_height])
    transport_pos = np.array([place_pos[0], place_pos[1], lift_height])
    place_target = np.asarray(place_pos, dtype=float)

    def segment(p_from, p_to):
        dist = np.linalg.norm(p_to - p_from)
        n = max(2, int(round(dist / velocity_mps * fps)))
        t = np.linspace(0, 1, n, endpoint=False)
        return p_from[None, :] + t[:, None] * (p_to - p_from)[None, :]

    lift_seg = segment(start_pos, lift_pos)
    transport_seg = segment(lift_pos, transport_pos)
    place_seg = segment(transport_pos, place_target)
    release_seg = np.repeat(place_target[None, :], release_frames, axis=0)

    positions = np.concatenate([lift_seg, transport_seg, place_seg, release_seg], axis=0)
    gripper_cmds = np.concatenate([
        np.full(len(lift_seg) + len(transport_seg) + len(place_seg), GRIPPER_ENGAGED),
        np.full(len(release_seg), GRIPPER_IDLE),
    ])
    stage = np.concatenate([
        np.full(len(lift_seg), STAGE_LIFT),
        np.full(len(transport_seg), STAGE_TRANSPORT),
        np.full(len(place_seg) + len(release_seg), STAGE_PLACE),
    ])
    quats_xyzw = np.repeat(ori_xyzw[None, :], len(positions), axis=0)

    return {
        "positions": positions,
        "quats_xyzw": quats_xyzw,
        "gripper_cmds": gripper_cmds,
        "stage": stage,
        "start_pos": start_pos,
    }


def step_robot(robot: TwinRobot, pos: np.ndarray, ori_xyzw: np.ndarray, gripper_cmd: float, n_steps: int) -> dict:
    """
    Like TwinRobot.move_to_target_state, but also surfaces the raw
    proprioceptive sensors (quat/joint) needed for the replay buffer --
    TwinRobot itself only exposes position, not orientation or joint state.

    robot0_joint_pos is inactive by default in this robosuite fork, but
    robot0_joint_pos_cos/_sin ARE active, so joint angles are recovered
    exactly via atan2 rather than left out.

    Gripper conversion is delegated to TwinRobot._convert_handgripper_pos_to_action
    itself (rather than reimplemented here) since its mapping differs by
    gripper_name (Robotiq85 width vs. the Schmalz ECBPiUR vacuum gripper's
    idle/engage command actually mounted on this rig) -- duplicating just one
    branch here would silently break the other gripper.
    """
    gripper_action = robot._convert_handgripper_pos_to_action(gripper_cmd)
    obs = robot.move_to_pose(pos, ori_xyzw, float(gripper_action), n_steps)

    robot_pos = obs["robot0_eef_pos"] - robot.robot_base_pos
    return {
        "robot_mask": np.squeeze(robot.get_robot_mask(obs)),
        "gripper_mask": np.squeeze(robot.get_gripper_mask(obs)),
        "rgb_img": robot.get_image(obs),
        "robot_pos": robot_pos,
        "robot_quat_xyzw": obs["robot0_eef_quat"],
        "joint_pos": np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"]),
        "joint_vel": obs["robot0_joint_vel"],
        "pos_err": float(np.linalg.norm(robot_pos - pos)),
    }


def run_episode(robot: TwinRobot, traj: dict, bg_frames: list, bg_idxs: np.ndarray,
                 writer: "ZarrReplayBufferWriter", video_path, fps: float) -> None:
    """
    Step the twin through one trajectory, composite each frame onto its
    sampled background, and append the episode to the shared zarr store.
    Frames with excessive tracking error are dropped from the episode
    entirely (not zero-padded) so every stored frame is trustworthy.
    """
    robot.reset()
    imgs = []
    actions, eef_poses, joint_poses, joint_vels, gripper_cmds, stages, timestamps = [], [], [], [], [], [], []
    num_skipped = 0
    dt = 1.0 / fps

    for idx in range(len(traj["positions"])):
        target_pos = traj["positions"][idx]
        target_ori = traj["quats_xyzw"][idx]
        target_gripper = traj["gripper_cmds"][idx]
        n_steps = robot.n_steps_long if idx == 0 else robot.n_steps_short

        result = step_robot(robot, target_pos, target_ori, target_gripper, n_steps)
        bg_frame = bg_frames[bg_idxs[idx]]

        if result["pos_err"] > TRACKING_ERROR_THRESHOLD:
            num_skipped += 1
            continue

        imgs.append(overlay_robot(bg_frame, result))
        actions.append(np.concatenate([target_pos, Rotation.from_quat(target_ori).as_rotvec()]))
        eef_poses.append(np.concatenate([result["robot_pos"], Rotation.from_quat(result["robot_quat_xyzw"]).as_rotvec()]))
        joint_poses.append(result["joint_pos"])
        joint_vels.append(result["joint_vel"])
        gripper_cmds.append(target_gripper)
        stages.append(traj["stage"][idx])
        timestamps.append(len(timestamps) * dt)

    if not imgs:
        print(f"  WARNING: episode produced 0 valid frames ({num_skipped} skipped), not writing it")
        return

    writer.add_episode({
        "action": np.stack(actions).astype(np.float64),
        "robot_eef_pose": np.stack(eef_poses).astype(np.float64),
        "robot_joint": np.stack(joint_poses).astype(np.float64),
        "robot_joint_vel": np.stack(joint_vels).astype(np.float64),
        "gripper_cmd": np.array(gripper_cmds, dtype=np.float64),
        "stage": np.array(stages, dtype=np.int64),
        "timestamp": np.array(timestamps, dtype=np.float64),
    })

    if video_path is not None:
        media.write_video(video_path, imgs, fps=fps)
    print(f"  {len(imgs)} frames written, {num_skipped} skipped (tracking error)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--background-video", required=True,
                         help="Path to a clean (robot/hand-free) plate video of the workspace")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "data", "processed", "synthetic_pickplace"))
    parser.add_argument("--demo-name", default="synthetic_pickplace")
    parser.add_argument("--robot", default="UR7e")
    parser.add_argument("--gripper", default="ECBPiUR",
                         help="Robosuite gripper name (this arm's actual tool is the Schmalz ECBPi UR "
                              "vacuum gripper; use 'Robotiq85' for a parallel-jaw twin instead)")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--region-low", type=float, nargs=3, default=[0.35, -0.25, 0.25], metavar=("X", "Y", "Z"),
                         help="Lower corner of the random start-pose region (robot-base frame, meters)")
    parser.add_argument("--region-high", type=float, nargs=3, default=[0.65, 0.25, 0.45], metavar=("X", "Y", "Z"))
    parser.add_argument("--place-pos", type=float, nargs=3, default=[0.5, 0.0, 0.25], metavar=("X", "Y", "Z"),
                         help="Fixed place position; override to your real place location")
    parser.add_argument("--lift-height", type=float, default=0.5,
                         help="Fixed Z the arm rises to before transporting (meters)")
    parser.add_argument("--yaw-range-deg", type=float, default=60.0,
                         help="+/- range for randomizing the start pose's yaw")
    parser.add_argument("--velocity", type=float, default=0.15, help="End-effector speed, m/s")
    parser.add_argument("--release-frames", type=int, default=10, help="Frames to hold while releasing the object")

    parser.add_argument("--hold-min", type=int, default=3, help="Min frames to hold a sampled background frame")
    parser.add_argument("--hold-max", type=int, default=10, help="Max frames to hold a sampled background frame")
    parser.add_argument("--max-bg-frames", type=int, default=1000, help="Cap on background frames loaded into memory")

    parser.add_argument("--fps", type=float, default=15.0, help="Output img/video frame rate")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True,
                         help="Also write a per-episode .mp4 next to the zarr store, for visual QA")
    parser.add_argument("--square", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    region_low, region_high = np.array(args.region_low), np.array(args.region_high)
    place_pos = np.array(args.place_pos)

    camera_params = build_real_camera_params(square=args.square)
    img_h, img_w = camera_params.resolution

    print(f"Reading background video: {args.background_video}")
    video = media.read_video(args.background_video)
    num_bg_frames = min(len(video), args.max_bg_frames)
    bg_frames = [np.asarray(video[i]) for i in range(num_bg_frames)]
    bg_frames = [cv2.resize(f, (img_w, img_h)) for f in bg_frames]
    if args.square:
        bg_frames = [center_crop_square(f) for f in bg_frames]

    print(f"Instantiating TwinRobot(robot_name='{args.robot}', gripper_name='{args.gripper}')...")
    robot = TwinRobot(
        robot_name=args.robot, gripper_name=args.gripper, camera_params=camera_params,
        camera_height=img_h, camera_width=img_w, render=args.render,
        n_steps_short=3, n_steps_long=75, square=args.square,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    zarr_path = os.path.join(args.output_dir, f"{args.demo_name}.zarr")
    writer = ZarrReplayBufferWriter(zarr_path)
    video_dir = os.path.join(args.output_dir, f"{args.demo_name}_videos")
    if args.save_videos:
        os.makedirs(video_dir, exist_ok=True)

    try:
        for episode_idx in range(args.num_episodes):
            rng = np.random.default_rng(args.seed + episode_idx)
            traj = generate_pickplace_trajectory(
                region_low, region_high, place_pos, args.lift_height, args.velocity,
                fps=args.fps, release_frames=args.release_frames, yaw_range_deg=args.yaw_range_deg, rng=rng,
            )
            bg_idxs = sample_background_indices(
                len(traj["positions"]), num_bg_frames, args.hold_min, args.hold_max, rng,
            )
            video_path = os.path.join(video_dir, f"{episode_idx}_{args.robot}.mp4") if args.save_videos else None
            print(f"Episode {episode_idx}: start_pos={traj['start_pos']}, {len(traj['positions'])} frames")
            run_episode(robot, traj, bg_frames, bg_idxs, writer, video_path, args.fps)
    finally:
        robot.close()

    total_frames = writer.meta["episode_ends"][-1] if len(writer.meta["episode_ends"]) else 0
    print(f"\nDone. Wrote {len(writer.meta['episode_ends'])} episodes ({total_frames} frames) to {zarr_path}")


if __name__ == "__main__":
    main()
