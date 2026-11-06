"""
Synthetic Pick-and-Place Processor Module

Generates synthetic pick-and-place demonstrations for pose randomization.
Unlike the other processors it does not consume recorded demos: each episode
samples a random start pose and a random pick spot, approaches and descends onto it,
engages suction, lifts, transports to a fixed place position, releases and retreats.
With `synthetic.object` set, a sim object is carried along (welded to the gripper
while suction is on, see phantom-robosuite's Phantom.set_suction). The trajectory is rendered with the MuJoCo twin robot and composited
onto frames sampled from a clean (robot/hand-free) workspace video.

A second, flange-mounted camera can be generated alongside the fixed one. It cannot
use a clean plate -- it moves with the robot -- so its background is rendered per
frame from a scene mosaic at the pose the robot actually reached; see
phantom.synthetic.wrist_background and phantom/scene_mosaic/README.md.

That background only exists where the scan went. The randomization region is therefore
checked against the mosaic before any episode is generated (and, with
synthetic.wrist.region_from_mosaic, derived from it), because a region the scan never
covered produces wrist frames that are part fill colour without anything failing --
see phantom.synthetic.wrist_region.

Output (under <processed_data_root_dir>/<demo_name>/synthetic_processor/):
    <demo_name>.zarr          diffusion_policy-style ReplayBuffer (see phantom.synthetic.replay_buffer)
    <demo_name>_videos/       per-episode QA videos (one per camera)
"""

import os
import json
import logging
from typing import Any, Dict, List

import mediapy as media
import numpy as np
from scipy.spatial.transform import Rotation

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths
from phantom.processors.robotinpaint_processor import RobotInpaintProcessor
from phantom.synthetic.background import load_background_frames, sample_background_indices
from phantom.synthetic.replay_buffer import ZarrReplayBufferWriter
from phantom.synthetic.trajectory import generate_pickplace_trajectory
from phantom.synthetic.wrist_background import WristMosaicBackground
from phantom.synthetic.twin_frames import command_to_flange, describe_tool_frames, measure_tool_frames
from phantom.synthetic.wrist_region import (describe_region, fit_region, region_coverage,
                                            scan_yaw_center)
from phantom.twin_robot import TwinRobot
from phantom.utils.image_utils import overlay_robot

logger = logging.getLogger(__name__)


class SyntheticPickPlaceProcessor(BaseProcessor):
    """
    Generates synthetic pick-and-place episodes with the twin robot overlaid on a clean plate.
    Run once per config (not per demo folder), see `process`.
    """
    TRACKING_ERROR_THRESHOLD = RobotInpaintProcessor.TRACKING_ERROR_THRESHOLD

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.synth = cfg.synthetic
        self.wrist = cfg.synthetic.get("wrist", None)
        self.twin_robot = None

    def _setup_paths_and_folders(self, cfg: Any) -> None:
        """No recorded demos are needed; only the output folder is created."""
        self.data_folder = None
        self.processed_data_folder = os.path.join(cfg.processed_data_root_dir, cfg.demo_name)
        os.makedirs(self.processed_data_folder, exist_ok=True)
        self.all_data_folders = []

    def _validate_config(self, cfg: Any) -> None:
        if cfg.input_resolution <= 0 or cfg.output_resolution <= 0:
            raise ValueError(f"Resolutions must be positive: input={cfg.input_resolution}, output={cfg.output_resolution}")
        if not os.path.exists(cfg.camera_intrinsics):
            raise FileNotFoundError(f"Camera intrinsics file not found: {cfg.camera_intrinsics}")
        if not os.path.exists(cfg.synthetic.background_video):
            raise FileNotFoundError(f"Background video not found: {cfg.synthetic.background_video}")
        wrist = cfg.synthetic.get("wrist", None)
        if wrist is not None and wrist.enabled:
            self._validate_wrist_model(wrist.mosaic_model)

    @staticmethod
    def _validate_wrist_model(model_dir: str) -> None:
        """
        Fail before the twin is built (which is slow) if the mosaic model directory is
        missing or incomplete. Its images are not in git, so a missing mosaic.png next
        to a present mosaic_meta.json is the common case and worth its own message.
        """
        required = ["mosaic.png", "mosaic_valid.png", "mosaic_meta.json"]
        meta_path = os.path.join(model_dir, "mosaic_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                gain_file = json.load(f).get("gain_file")
            if gain_file:  # loaded whenever the meta names it, apply_gain or not
                required.append(gain_file)
        missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
        if not missing:
            return
        raise FileNotFoundError(
            f"Wrist mosaic model at {model_dir!r} is missing {', '.join(missing)}. "
            "Build one with `python -m phantom.scene_mosaic.build_mosaic --data <scan> "
            "--episode <n> --camera <n> --out " + str(model_dir) + "`, or copy an "
            "existing model directory there (phantom/scene_mosaic/README.md); set "
            "synthetic.wrist.enabled=false to generate the fixed camera only. Note "
            "the path is relative to the working directory, which process_data.py is "
            "normally run from phantom/."
        )

    def process_one_demo(self, data_sub_folder: str) -> None:
        # Synthetic generation is not tied to a demo folder.
        self.process()

    def process(self) -> None:
        """Generate cfg.synthetic.num_episodes episodes into a single zarr store."""
        synth = self.synth
        paths = Paths(data_path=self.processed_data_folder, robot_name=self.robot)
        out_dir = str(paths.synthetic_processor)
        os.makedirs(out_dir, exist_ok=True)
        zarr_path = os.path.join(out_dir, f"{self.cfg.demo_name}.zarr")
        video_dir = os.path.join(out_dir, f"{self.cfg.demo_name}_videos")
        if synth.save_videos:
            os.makedirs(video_dir, exist_ok=True)

        camera_params = self._get_mujoco_camera_params()
        img_h, img_w = camera_params.resolution
        print(f"Reading background video: {synth.background_video}")
        bg_frames = load_background_frames(synth.background_video, synth.max_bg_frames, img_w, img_h, self.square)

        # The wrist camera is configured from the same mosaic its backgrounds are
        # rendered from, so the two cannot disagree (phantom/scene_mosaic/README.md).
        wrist_bg = None
        wrist_camera_params = None
        if self.wrist is not None and self.wrist.enabled:
            wrist_bg = WristMosaicBackground(
                self.wrist.mosaic_model, img_w, img_h, self.square,
                apply_distortion=self.wrist.apply_distortion,
                apply_gain=self.wrist.apply_gain,
                fill_color=self.wrist.fill_color,
                mount_translation=self.wrist.mount_translation,
            )
            wrist_camera_params = wrist_bg.camera_params()
            print(wrist_bg.describe())

        region_low, region_high = np.array(synth.region_low), np.array(synth.region_high)
        place_pos = np.array(synth.place_pos)
        lift_height = float(synth.lift_height)
        yaw_center_deg = float(synth.get("yaw_center_deg", 0.0))

        print(f"Instantiating TwinRobot(robot_name='{self.robot}', gripper_name='{self.gripper}')...")
        self.twin_robot = TwinRobot(
            self.robot, self.gripper, camera_params,
            camera_height=img_h, camera_width=img_w, render=self.render,
            n_steps_short=3, n_steps_long=75, square=self.square,
            wrist_camera_params=wrist_camera_params,
            show_wrist_camera=bool(wrist_bg is not None and self.wrist.show_camera_body),
            grasp_object=self._grasp_object_cfg(),
        )
        # Measured, not hardcoded: both change with the gripper (phantom/synthetic/twin_frames.py).
        # Needed for flange-frame actions and for the wrist region check.
        self.T_flange_grip, self.ori_correction = measure_tool_frames(self.twin_robot)
        print(f"  {describe_tool_frames(self.T_flange_grip, self.ori_correction)}")

        if wrist_bg is not None:
            region_low, region_high, lift_height, place_pos, yaw_center_deg = self._check_wrist_region(
                wrist_bg, region_low, region_high, place_pos, lift_height, yaw_center_deg)
        writer = ZarrReplayBufferWriter(zarr_path)

        try:
            for episode_idx in range(synth.num_episodes):
                rng = np.random.default_rng(synth.seed + episode_idx)
                traj = generate_pickplace_trajectory(
                    region_low, region_high, place_pos, lift_height, synth.velocity,
                    fps=synth.fps, release_frames=synth.release_frames,
                    yaw_range_deg=synth.yaw_range_deg, rng=rng, yaw_center_deg=yaw_center_deg,
                    grasp_frames=int(synth.get("grasp_frames", 8)),
                    approach_height=synth.get("approach_height", None),
                    settle_frames=int(synth.get("settle_frames", 3)),
                )
                bg_idxs = sample_background_indices(
                    len(traj["positions"]), len(bg_frames), synth.hold_min, synth.hold_max, rng,
                )
                video_path = wrist_video_path = None
                if synth.save_videos:
                    video_path = os.path.join(video_dir, f"{episode_idx}_{self.robot}.mp4")
                    if wrist_bg is not None:
                        wrist_video_path = os.path.join(video_dir, f"{episode_idx}_{self.robot}_wrist.mp4")
                print(f"Episode {episode_idx}: start_pos={traj['start_pos']}, {len(traj['positions'])} frames")
                self._run_episode(traj, bg_frames, bg_idxs, writer, video_path,
                                  wrist_bg=wrist_bg, wrist_video_path=wrist_video_path)
        finally:
            self.twin_robot.close()
            self.twin_robot = None

        total_frames = writer.meta["episode_ends"][-1] if len(writer.meta["episode_ends"]) else 0
        print(f"\nDone. Wrote {len(writer.meta['episode_ends'])} episodes ({total_frames} frames) to {zarr_path}")

    def _grasp_object_cfg(self):
        """
        The plain-dict `synthetic.object` config, or None for a motion-only run.
        Suction carries it, so any other gripper gets a warning (it would still be
        welded on, which is not what that gripper does).
        """
        obj = self.synth.get("object", None)
        if obj is None:
            return None
        if not str(self.gripper).startswith("ECBPi"):
            print(f"  WARNING: synthetic.object is carried by suction but gripper is {self.gripper!r}; "
                  "set gripper=ECBPiUR for a physically meaningful grasp.")
        return {"type": obj.get("type", "box"), "size": list(obj.size), "rgba": list(obj.rgba)}

    def _check_wrist_region(self, wrist_bg, region_low, region_high, place_pos, lift_height,
                            yaw_center_deg):
        """
        Confirm the randomization region is one the mosaic can actually back, before
        spending minutes generating episodes against a background that is half fill
        colour. Returns the region to generate with, which is the configured one unless
        `synthetic.wrist.region_from_mosaic` asked for the mosaic's own.

        Below `synthetic.wrist.min_region_coverage` the configured region is refused
        rather than warned about: coverage this low is not a quality setting, it means
        the trajectories leave the scanned workspace, and every frame generated that
        way has to be thrown out anyway.
        """
        T_fg, C = self.T_flange_grip, self.ori_correction
        b = wrist_bg.scan_bounds()
        print(f"  scan covered flange xyz {np.round(b['flange_low'], 3).tolist()} .. "
              f"{np.round(b['flange_high'], 3).tolist()} ({b['n_dwells']} dwells), "
              f"camera at z {b['camera_z_low']:.3f} over a plane at z {b['plane_z']:.3f}")

        scan_yaw, yaw_residual = scan_yaw_center(wrist_bg, C)
        print(f"  scan orientation corresponds to trajectory yaw {scan_yaw:+.1f} deg "
              f"(residual {yaw_residual:.1f} deg); configured yaw band is "
              f"{yaw_center_deg:+.1f} +- {self.synth.yaw_range_deg:.0f} deg")

        floor = float(self.wrist.get("min_region_coverage", 0.9))
        cfg_cov = region_coverage(wrist_bg, T_fg, region_low, region_high, place_pos,
                                  lift_height, self.synth.yaw_range_deg, ori_correction=C,
                                  yaw_center_deg=yaw_center_deg)
        print(f"  {describe_region('configured region', region_low, region_high, lift_height, place_pos)}")
        print(f"  -> wrist coverage over that region: worst {cfg_cov['min'] * 100:.1f}%, "
              f"mean {cfg_cov['mean'] * 100:.1f}% ({cfg_cov['n_poses']} probe poses)")

        if self.wrist.get("region_from_mosaic", False):
            fit = fit_region(wrist_bg, T_fg, self.synth.yaw_range_deg, min_coverage=floor,
                             z_offset=float(self.wrist.get("pick_height_above_scan", 0.0)),
                             ori_correction=C)
            print(f"  {describe_region('region_from_mosaic', fit['region_low'], fit['region_high'], fit['lift_height'], fit['place_pos'])}"
                  f", yaw {fit['yaw_center_deg']:+.1f} +- {self.synth.yaw_range_deg:.0f} deg")
            print(f"  -> wrist coverage over that region: worst {fit['coverage']['min'] * 100:.1f}%, "
                  f"mean {fit['coverage']['mean'] * 100:.1f}%")
            # A single-height scan buys range in xy cheaply but height expensively: the
            # camera's footprint grows as it rises, so lifting runs off the scanned
            # patch long before translating does. If no lift survives the floor, the
            # "pick and place" is a slide, which is worth saying out loud.
            lift_gain = fit['lift_height'] - float(fit['region_low'][2])
            if lift_gain < 0.02:
                # print, not logger.warning: process_data.py runs under hydra, which
                # takes over logging configuration, and these warnings never reach the
                # console -- which is how a run can look fine and still be unusable.
                print(f"  WARNING: the mosaic supports almost no lift at min_region_coverage "
                      f"{floor * 100:.0f}% (fitted lift {lift_gain * 1000:.0f} mm), so the "
                      "trajectories stay in the picking plane. Lower min_region_coverage to "
                      "trade coverage for lift, or rebuild the mosaic from a scan flown at "
                      "more than one height.")
            if fit['coverage']['min'] < floor:
                raise ValueError(
                    f"Even the mosaic's own region only reaches {fit['coverage']['min'] * 100:.1f}% "
                    f"coverage (< min_region_coverage {floor * 100:.1f}%). The scan is too small "
                    "for any useful randomization at this camera height -- extend it and rebuild "
                    "the mosaic (phantom/scene_mosaic/README.md)."
                )
            return (fit['region_low'], fit['region_high'], fit['lift_height'], fit['place_pos'],
                    fit['yaw_center_deg'])

        if cfg_cov['min'] < floor:
            fit = fit_region(wrist_bg, T_fg, self.synth.yaw_range_deg, min_coverage=floor,
                             ori_correction=C)
            raise ValueError(
                f"The configured randomization region leaves the scanned workspace: wrist "
                f"coverage falls to {cfg_cov['min'] * 100:.1f}% (< min_region_coverage "
                f"{floor * 100:.1f}%) at grip-site pose "
                f"{np.round(cfg_cov['worst_pos'], 3).tolist()}, so those frames would show "
                "unscanned workspace as fill_color. Note the region is in grip-site "
                "coordinates while the scan is in flange coordinates, one tool length "
                f"({np.linalg.norm(T_fg[:3, 3]):.3f} m) apart. Either set "
                "synthetic.wrist.region_from_mosaic=true, or use "
                f"region_low={np.round(fit['region_low'], 3).tolist()}, "
                f"region_high={np.round(fit['region_high'], 3).tolist()}, "
                f"lift_height={fit['lift_height']:.3f}, "
                f"place_pos={np.round(fit['place_pos'], 3).tolist()}, "
                f"yaw_center_deg={fit['yaw_center_deg']:.1f}; or lower "
                "synthetic.wrist.min_region_coverage to accept partly uncovered frames."
            )
        return region_low, region_high, lift_height, place_pos, yaw_center_deg

    def _step_robot(self, pos: np.ndarray, ori_xyzw: np.ndarray, gripper_cmd: float, n_steps: int) -> Dict[str, Any]:
        """
        Like TwinRobot.move_to_target_state, but also surfaces the raw
        proprioceptive sensors (quat/joint) needed for the replay buffer --
        TwinRobot itself only exposes position, not orientation or joint state.

        robot0_joint_pos is inactive by default in this robosuite fork, but
        robot0_joint_pos_cos/_sin ARE active, so joint angles are recovered
        exactly via atan2 rather than left out.

        Gripper conversion is delegated to TwinRobot._convert_handgripper_pos_to_action
        (rather than reimplemented) since its mapping differs by gripper_name
        (Robotiq85 width vs. the Schmalz ECBPiUR vacuum gripper's idle/engage command).
        """
        robot = self.twin_robot
        gripper_action = robot._convert_handgripper_pos_to_action(gripper_cmd)
        obs = robot.move_to_pose(pos, ori_xyzw, float(gripper_action), n_steps)

        robot_pos = obs["robot0_eef_pos"] - robot.robot_base_pos
        result = {
            "robot_mask": np.squeeze(robot.get_robot_mask(obs)),
            "gripper_mask": np.squeeze(robot.get_gripper_mask(obs)),
            "rgb_img": robot.get_image(obs),
            "robot_pos": robot_pos,
            "robot_quat_xyzw": obs["robot0_eef_quat"],
            "joint_pos": np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"]),
            "joint_vel": obs["robot0_joint_vel"],
            "pos_err": float(np.linalg.norm(robot_pos - pos)),
        }
        if robot.has_grasp_object:
            result["object_mask"] = np.squeeze(robot.get_object_mask(obs))
            result["object_pose"] = robot.get_object_pose()
        if robot.wrist_camera_name:
            cam = robot.wrist_camera_name
            result["wrist_rgb_img"] = robot.get_cam_image(obs, cam)
            result["wrist_robot_mask"] = np.squeeze(robot.get_robot_mask(obs, cam))
            result["wrist_gripper_mask"] = np.squeeze(robot.get_gripper_mask(obs, cam))
            if robot.has_grasp_object:
                result["wrist_object_mask"] = np.squeeze(robot.get_object_mask(obs, cam))
            # Read back rather than recomputed from robot_pos/robot_quat_xyzw: those
            # describe the grip site, not the flange body the camera hangs off.
            result["wrist_cam_pose"] = robot.get_wrist_camera_pose()
        result["flange_pose"] = robot.get_flange_pose()
        return result

    def _eef_pose6(self, result: Dict[str, Any]) -> np.ndarray:
        """
        The [x, y, z, rx, ry, rz] written to the buffer's `robot_eef_pose`, in the frame
        `synthetic.eef_pose_frame` names.

        The default is the flange, because that is what the real robot's recordings in
        this stack report -- `phantom.scene_mosaic.build_mosaic` composes
        `robot_eef_pose` directly with the hand-eye flange -> camera transform, so a
        synthetic buffer written in grip-site coordinates would be offset from real
        data by the tool length (0.155 m for a Robotiq85) and a policy trained across
        both would see two different conventions for one field.

        "grip" keeps the frame TwinRobot tracks (`robot0_eef_pos/quat`), for datasets
        that were generated that way before this was configurable.
        """
        frame = self.synth.get("eef_pose_frame", "flange")
        if frame == "flange":
            T = result["flange_pose"]
            return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])
        if frame == "grip":
            return np.concatenate([result["robot_pos"],
                                   Rotation.from_quat(result["robot_quat_xyzw"]).as_rotvec()])
        raise ValueError(f"synthetic.eef_pose_frame must be 'flange' or 'grip', got {frame!r}")

    def _action_pose6(self, target_pos: np.ndarray, target_ori_xyzw: np.ndarray) -> np.ndarray:
        """
        The [x, y, z, rx, ry, rz] written to the buffer's `action`: the commanded pose, in
        the same frame as `robot_eef_pose` (`synthetic.eef_pose_frame`), so the two fields
        never disagree on convention.

        For "flange" the grip-site command is converted to the flange pose the arm settles
        at for it -- one tool length back along the tool axis and with the controller's
        commanded-vs-settled twist removed -- which is what a real robot with its TCP at
        the flange (the UR default) is commanded. "grip" writes the raw grip-site command,
        as datasets made before this did.
        """
        frame = self.synth.get("eef_pose_frame", "flange")
        if frame == "flange":
            return command_to_flange(target_pos, target_ori_xyzw, self.T_flange_grip, self.ori_correction)
        if frame == "grip":
            return np.concatenate([target_pos, Rotation.from_quat(target_ori_xyzw).as_rotvec()])
        raise ValueError(f"synthetic.eef_pose_frame must be 'flange' or 'grip', got {frame!r}")

    def _run_episode(self, traj: Dict[str, Any], bg_frames: List[np.ndarray], bg_idxs: np.ndarray,
                     writer: ZarrReplayBufferWriter, video_path,
                     wrist_bg=None, wrist_video_path=None) -> None:
        """
        Step the twin through one trajectory, composite each frame onto its
        sampled background, and append the episode to the shared zarr store.
        Frames with excessive tracking error are dropped entirely (not
        zero-padded) so every stored frame is trustworthy.

        With a wrist camera, each frame gets a second composite: its background is
        rendered from the mosaic at the camera pose MuJoCo actually rendered that
        frame's robot from, so the two halves of the composite are the same camera by
        construction -- tracking error included.
        """
        self.twin_robot.reset()
        has_object = self.twin_robot.has_grasp_object
        if has_object:
            # Rest the object's centre half its height below the pick point, so the cup
            # lands on its top face.
            half_h = float(self.synth.object.size[2 if self.synth.object.get("type", "box") == "box" else 1])
            self.twin_robot.place_object(traj["pick_pos"] - np.array([0.0, 0.0, half_h]),
                                         traj["object_quat_xyzw"])
        imgs = []
        wrist_imgs, wrist_coverage = [], []
        actions, eef_poses, joint_poses, joint_vels, gripper_cmds, stages, timestamps = [], [], [], [], [], [], []
        object_poses, suctions = [], []
        num_skipped = 0
        dt = 1.0 / self.synth.fps

        for idx in range(len(traj["positions"])):
            target_pos = traj["positions"][idx]
            target_ori = traj["quats_xyzw"][idx]
            target_gripper = traj["gripper_cmds"][idx]
            n_steps = self.twin_robot.n_steps_long if idx == 0 else self.twin_robot.n_steps_short

            # Driven from the trajectory, not from kept frames: a frame dropped below for
            # tracking error must not swallow the grasp or the release.
            if has_object:
                self.twin_robot.set_suction(bool(traj["suction"][idx]))
            result = self._step_robot(target_pos, target_ori, target_gripper, n_steps)
            if result["pos_err"] > self.TRACKING_ERROR_THRESHOLD:
                num_skipped += 1
                continue

            eef_pose6 = self._eef_pose6(result)

            imgs.append(overlay_robot(bg_frames[bg_idxs[idx]], result))
            if wrist_bg is not None:
                wrist_frame, coverage = wrist_bg.render_at_camera(result["wrist_cam_pose"])
                wrist_imgs.append(overlay_robot(wrist_frame, {
                    "rgb_img": result["wrist_rgb_img"],
                    "robot_mask": result["wrist_robot_mask"],
                    "gripper_mask": result["wrist_gripper_mask"],
                    "object_mask": result.get("wrist_object_mask"),
                }))
                wrist_coverage.append(coverage)

            actions.append(self._action_pose6(target_pos, target_ori))
            eef_poses.append(eef_pose6)
            joint_poses.append(result["joint_pos"])
            joint_vels.append(result["joint_vel"])
            gripper_cmds.append(target_gripper)
            suctions.append(bool(traj["suction"][idx]))
            if has_object:
                object_poses.append(result["object_pose"])
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
            "suction": np.array(suctions, dtype=bool),
            **({"object_pose": np.stack(object_poses).astype(np.float64)} if has_object else {}),
            "timestamp": np.array(timestamps, dtype=np.float64),
        })

        if video_path is not None:
            media.write_video(video_path, imgs, fps=self.synth.fps)
        if wrist_video_path is not None and wrist_imgs:
            media.write_video(wrist_video_path, wrist_imgs, fps=self.synth.fps)
        print(f"  {len(imgs)} frames written, {num_skipped} skipped (tracking error)")

        if wrist_coverage:
            cov = np.array(wrist_coverage)
            print(f"  wrist mosaic coverage: mean {cov.mean() * 100:.1f}%, "
                  f"worst {cov.min() * 100:.1f}%")
            # Warning only: the frames are kept, since dropping them would punch holes
            # in the trajectory the proprioception describes. Low coverage means the
            # randomization region sends the wrist camera outside the scanned area --
            # either shrink the region or extend the scan.
            if cov.min() < self.wrist.min_coverage:
                print(f"  WARNING: wrist background coverage fell to {cov.min() * 100:.1f}% "
                      f"(< min_coverage {self.wrist.min_coverage * 100:.1f}%): those frames "
                      "show unscanned workspace as fill_color")
