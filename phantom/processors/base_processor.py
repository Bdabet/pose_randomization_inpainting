import os
import json
import logging
import numpy as np
import shutil
import errno
from typing import Tuple, TYPE_CHECKING
from pathlib import Path
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from phantom.utils.data_utils import get_parent_folder_of_package
from phantom.utils.image_utils import (get_intrinsics_from_json,
                                       get_transformation_matrix_from_extrinsics,
                                       normalize_extrinsics)
from phantom.processors.paths import Paths, PathsConfig

if TYPE_CHECKING:
    from phantom.twin_bimanual_robot import MujocoCameraParams

logger = logging.getLogger(__name__)

class BaseProcessor: 
    def __init__(self, cfg: DictConfig): 
        # Store configuration for potential future use
        self.cfg = cfg
        
        # Apply configuration to instance attributes
        self._apply_config(cfg)
        
        # Validate configuration
        self._validate_config(cfg)
        
        # Set up paths and data folders
        self._setup_paths_and_folders(cfg)
        
        # Initialize camera parameters
        self._init_camera_parameters()

    def _apply_config(self, cfg: DictConfig) -> None:
        """Apply configuration to instance attributes."""
        # Basic attributes
        self.input_resolution = cfg.input_resolution
        self.output_resolution = cfg.output_resolution
        self.project_folder = get_parent_folder_of_package("phantom")
        self.debug = cfg.debug
        self.n_processes = cfg.n_processes
        self.verbose = cfg.verbose
        self.skip_existing = cfg.skip_existing
        self.robot = cfg.robot
        self.gripper = cfg.gripper
        self.square = cfg.square
        self.epic = cfg.epic
        self.bimanual_setup = cfg.bimanual_setup
        self.target_hand = cfg.target_hand
        self.constrained_hand = cfg.constrained_hand
        self.depth_for_overlay = cfg.depth_for_overlay
        self.render = cfg.render
        self.debug_cameras = getattr(cfg, 'debug_cameras', [])
        
        # Apply bimanual setup logic
        if self.bimanual_setup != "single_arm":
            self.target_hand = "both"

    def _validate_config(self, cfg: DictConfig) -> None:
        """Validate critical configuration parameters."""
        if cfg.input_resolution <= 0 or cfg.output_resolution <= 0:
            raise ValueError(f"Resolutions must be positive: input={cfg.input_resolution}, output={cfg.output_resolution}")
        
        if not os.path.exists(cfg.data_root_dir):
            raise FileNotFoundError(f"Data root directory not found: {cfg.data_root_dir}")
        
        if not os.path.exists(cfg.camera_intrinsics):
            raise FileNotFoundError(f"Camera intrinsics file not found: {cfg.camera_intrinsics}")

    def _setup_paths_and_folders(self, cfg: DictConfig) -> None:
        """Set up paths configuration and create necessary directories."""
        # Set up paths configuration
        self.paths_config = PathsConfig()
        self.paths_config.config['data_root'] = cfg.data_root_dir
        self.paths_config.config['processed_root'] = cfg.processed_data_root_dir
        
        self.data_folder = os.path.join(cfg.data_root_dir, cfg.demo_name)
        self.processed_data_folder = os.path.join(cfg.processed_data_root_dir, cfg.demo_name)
        
        # Validate that data folder exists
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            
        os.makedirs(self.processed_data_folder, exist_ok=True)

        # Get all folders in data_folder
        try:
            all_data_folders = [d1 for d1 in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, d1))]
            self.all_data_folders = sorted(all_data_folders, key=lambda x: int(x))
            self.all_data_folders_idx = {x: idx for idx, x in enumerate(self.all_data_folders)}
        except OSError as e:
            if e.errno == errno.EACCES:
                raise PermissionError(f"Permission denied accessing data folder: {self.data_folder}")
            elif e.errno == errno.ENOENT:
                raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            else:
                raise RuntimeError(f"OS error accessing data folder {self.data_folder}: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid folder name format in {self.data_folder}. Folders should be numbered: {e}")

    def _init_camera_parameters(self) -> None:
        """Initialize camera intrinsics and extrinsics."""
        # Get camera intrinsics and extrinsics
        self.intrinsics_dict, self.intrinsics_matrix = self.get_intrinsics(self.cfg.camera_intrinsics)

        # Use camera_extrinsics from config if available, otherwise determine from bimanual_setup
        if hasattr(self.cfg, 'camera_extrinsics') and self.cfg.camera_extrinsics:
            camera_extrinsics_path = self.cfg.camera_extrinsics
        else:
            camera_extrinsics_path = self._get_camera_extrinsics_path()
            
        self.T_cam2robot, self.extrinsics = self.get_extrinsics(camera_extrinsics_path)

    def _get_camera_extrinsics_path(self) -> str:
        """Get the appropriate camera extrinsics path based on bimanual setup."""
        if self.bimanual_setup == "shoulders":
            return "camera/camera_extrinsics_ego_bimanual_shoulders.json"
        elif self.bimanual_setup == "single_arm":
            return "camera/camera_extrinsics.json"
        else:
            raise ValueError(f"Invalid bimanual setup: {self.bimanual_setup}. Must be 'single_arm' or 'shoulders'.")
    
    def get_paths(self, data_path: str) -> Paths:
        """
        Get all file paths for a demo.
        
        Args:
            data_path: Path to the demo data
            
        Returns:
            Paths object containing all file paths
        """
        paths = Paths(
            data_path=Path(data_path),
            robot_name=self.robot
        )
        paths.ensure_directories_exist()
        return paths
    
    def get_save_folder(self, data_sub_folder: str) -> str:
        data_sub_folder_fullpath = os.path.join(self.data_folder, str(data_sub_folder))
        save_folder = os.path.join(self.processed_data_folder, str(data_sub_folder))
        # Check existing dirs using os.scandir
        with os.scandir(self.processed_data_folder) as it:
            existing_dirs = {entry.name for entry in it if entry.is_dir()}
        if str(data_sub_folder) not in existing_dirs:
            shutil.copytree(data_sub_folder_fullpath, save_folder)
        return save_folder
    
    def process_one_demo(self, data_sub_folder: str): 
        raise NotImplementedError

    def get_intrinsics(self, intrinsics_path: str) -> Tuple[dict, np.ndarray]:
        intrinsics_matrix, intrinsics_dict = get_intrinsics_from_json(intrinsics_path)
        if self.square:
            intrinsics_dict, intrinsics_matrix = self.update_intrinsics_for_square_image(self.input_resolution,
                                                                                        intrinsics_dict, 
                                                                                        intrinsics_matrix)
        return intrinsics_dict, intrinsics_matrix
    
    def get_extrinsics(self, extrinsics_path: str) -> Tuple[np.ndarray, dict]:
        """Load and process camera extrinsics from JSON file.
        
        Args:
            extrinsics_path: Path to the extrinsics JSON file
            
        Returns:
            Tuple of (transformation_matrix, extrinsics_dict)
            
        Raises:
            FileNotFoundError: If extrinsics file doesn't exist
            json.JSONDecodeError: If extrinsics file is invalid JSON
            ValueError: If extrinsics data is invalid
        """
        if not os.path.exists(extrinsics_path):
            raise FileNotFoundError(f"Camera extrinsics file not found: {extrinsics_path}")
            
        try:
            with open(extrinsics_path, "r") as f:
                camera_extrinsics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in extrinsics file {extrinsics_path}: {str(e)}")
        
        try:
            camera_extrinsics = normalize_extrinsics(camera_extrinsics)
            T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)
        except Exception as e:
            raise ValueError(f"Failed to process extrinsics data from {extrinsics_path}: {str(e)}")
            
        return T_cam2robot, camera_extrinsics

    @staticmethod
    def update_intrinsics_for_square_image(img_h: int, intrinsics_dict: dict, 
                                           intrinsics_matrix: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Adjusts camera intrinsic parameters for a square image by modifying the principal point offset.

        Args:
            img_h (int): Height of the image (assumed to be square).
            intrinsics_dict (dict): Dictionary of intrinsic parameters.
            intrinsics_matrix (np.ndarray): Intrinsic matrix.

        Returns:
            Tuple[dict, np.ndarray]: Updated intrinsic parameters and matrix.
        """
        img_w = img_h * 16 // 9
        offset = (img_w - img_h) // 2
        intrinsics_dict["cx"] -= offset
        intrinsics_matrix[0, 2] -= offset
        return intrinsics_dict, intrinsics_matrix

    def _get_mujoco_camera_params(self) -> "MujocoCameraParams":
        """
        Generate MuJoCo camera parameters from real-world camera calibration.
        
        Returns:
            MujocoCameraParams object with calibrated camera settings
        """
        # Imported lazily: pulls in robosuite/robomimic, which most processors don't need
        from phantom.twin_bimanual_robot import MujocoCameraParams

        # Extract real-world camera extrinsics and convert to MuJoCo format
        extrinsics = self.extrinsics[0]
        camera_ori_wxyz = self._convert_real_camera_ori_to_mujoco(
            np.array(extrinsics["camera_base_ori"])
        )

        # Calculate image dimensions and camera intrinsics
        img_w, img_h = self._get_image_dimensions()
        offset = self._calculate_image_offset(img_w, img_h)
        fx, fy, cx, cy = self._get_camera_intrinsics(offset)
        sensor_width, sensor_height = self._calculate_sensor_size(img_w, img_h, fx, fy)

        # Select appropriate camera name based on dataset
        if self.epic:
            camera_name = "zed"
        else:
            camera_name = "frontview"
            
        return MujocoCameraParams(
            name=camera_name,
            pos=extrinsics["camera_base_pos"],
            ori_wxyz=camera_ori_wxyz,
            fov=self.intrinsics_dict["v_fov"],
            resolution=(img_h, img_w),
            sensorsize=np.array([sensor_width, sensor_height]),
            principalpixel=np.array([img_w/2-cx, cy-img_h/2]),
            focalpixel=np.array([fx, fy])
        )
    
    def _get_image_dimensions(self) -> Tuple[int, int]:
        """
        Calculate image dimensions based on input resolution configuration.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        # Epic
        if self.input_resolution == 256:
            img_w = 456
        # Phantom paper (1080) and any other 16:9 height, e.g. 720 for a 1280x720 camera
        else:
            img_w = self.input_resolution * 16 // 9
        img_h = self.input_resolution
        return img_w, img_h
    
    def _calculate_image_offset(self, img_w: int, img_h: int) -> int:
        """
        Calculate horizontal image offset for square aspect ratio processing.
        
        Args:
            img_w: Image width in pixels
            img_h: Image height in pixels
            
        Returns:
            Horizontal offset in pixels
        """
        if self.square:
            offset = (img_w - img_h) // 2
        else:
            offset = 0
        return offset
    
    def _get_camera_intrinsics(self, offset: int) -> Tuple[float, float, float, float]:
        """
        Extract camera intrinsic parameters with offset correction.
        
        Args:
            offset: Horizontal offset for principal point adjustment
            
        Returns:
            Tuple of (fx, fy, cx, cy) camera intrinsic parameters
        """
        return self.intrinsics_dict["fx"], self.intrinsics_dict["fy"], self.intrinsics_dict["cx"]+offset, self.intrinsics_dict["cy"]

    def _calculate_sensor_size(self, img_w: int, img_h: int, fx: float, fy: float) -> Tuple[float, float]:
        """
        Calculate physical sensor dimensions from image resolution and focal length.
        
        Args:
            img_w: Image width in pixels
            img_h: Image height in pixels
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            
        Returns:
            Tuple of (sensor_width, sensor_height) in meters
        """
        sensor_width = img_w / fy / 1000
        sensor_height = img_h / fx / 1000
        return sensor_width, sensor_height
    
    @staticmethod
    def _convert_real_camera_ori_to_mujoco(camera_ori_matrix: np.ndarray) -> np.ndarray:
        """
        Convert real-world camera orientation to MuJoCo coordinate system.
        
        Args:
            camera_ori_matrix: 3x3 rotation matrix in real-world coordinates
            
        Returns:
            Quaternion in WXYZ format for MuJoCo
        """
        # Apply coordinate system transformation (flip Y and Z axes)
        camera_ori_matrix[:, [1, 2]] = -camera_ori_matrix[:, [1, 2]]
        
        # Convert to quaternion in MuJoCo's WXYZ format
        r = Rotation.from_matrix(camera_ori_matrix)
        camera_ori_wxyz = r.as_quat(scalar_first=True)
        return camera_ori_wxyz
