import logging
from enum import Enum
from tqdm import tqdm
from joblib import Parallel, delayed  # type: ignore
import hydra
from omegaconf import DictConfig

from phantom.processors.base_processor import BaseProcessor

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

class ProcessingMode(Enum):
    """Enumeration of valid processing modes."""
    BBOX = "bbox"
    HAND2D = "hand2d"
    HAND3D = "hand3d"
    HAND_SEGMENTATION = "hand_segmentation"
    ARM_SEGMENTATION = "arm_segmentation"
    ACTION = "action"
    SMOOTHING = "smoothing"
    HAND_INPAINT = "hand_inpaint"
    ROBOT_INPAINT = "robot_inpaint"
    SYNTHETIC_PICKPLACE = "synthetic_pickplace"
    ALL = "all"

PROCESSING_ORDER = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "hand_segmentation",
    "hand3d",
    "action",
    "smoothing",
    "hand_inpaint",
    "robot_inpaint",
]

PROCESSING_ORDER_EPIC = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "action",
    "smoothing",
    "hand_inpaint",
    "robot_inpaint",
]

# Not part of "all": generates data from scratch instead of processing recorded demos,
# so it must be requested explicitly and runs once rather than once per demo folder.
SYNTHETIC_MODE = "synthetic_pickplace"

def select_modes(cfg: DictConfig, synthetic: bool = False) -> list:
    """
    Resolve cfg.mode (str, comma-separated str, or list) into an ordered list of modes to run.
    The per-demo modes are returned by default; synthetic=True returns only the run-once synthetic mode.
    """
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    valid_modes = [SYNTHETIC_MODE] if synthetic else processing_order

    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all" and not synthetic:
                    selected_modes.extend(processing_order)
                elif mode in valid_modes:
                    selected_modes.append(mode)
            return selected_modes
        return [m for m in valid_modes if m == cfg.mode or (cfg.mode == "all" and not synthetic)]

    # For list of modes, use the order provided by user
    selected_modes = []
    for mode in cfg.mode:
        if mode == "all" and not synthetic:
            selected_modes.extend(processing_order)
        elif mode in valid_modes:
            selected_modes.append(mode)
    return selected_modes

def process_synthetic(cfg: DictConfig, processor_classes: dict) -> None:
    print(f"----------------- {SYNTHETIC_MODE.upper()} PROCESSOR -----------------")
    processor_classes[SYNTHETIC_MODE](cfg).process()

def process_one_demo(data_sub_folder: str, cfg: DictConfig, processor_classes: dict) -> None:
    selected_modes = select_modes(cfg)
    
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        try:
            processor.process_one_demo(data_sub_folder)
        except Exception as e:
            print(f"Error in {mode} processing: {e}")
            if cfg.debug:
                raise

def process_all_demos(cfg: DictConfig, processor_classes: dict) -> None:
    selected_modes = select_modes(cfg)
    
    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        for data_sub_folder in tqdm(all_data_folders):
            try:
                processor.process_one_demo(data_sub_folder)
            except Exception as e:
                print(f"Error in {mode} processing: {e}")
                if cfg.debug:
                    raise

def process_all_demos_parallel(cfg: DictConfig, processor_classes: dict) -> None:
    selected_modes = select_modes(cfg)
    
    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg) 
        Parallel(n_jobs=cfg.n_processes)(
            delayed(processor.process_one_demo)(data_sub_folder) for data_sub_folder in all_data_folders
        )

def get_processor_classes(cfg: DictConfig) -> dict:
    """Initialize the processor classes"""
    from phantom.processors.bbox_processor import BBoxProcessor
    from phantom.processors.segmentation_processor import HandSegmentationProcessor, ArmSegmentationProcessor
    from phantom.processors.hand_processor import Hand2DProcessor, Hand3DProcessor
    from phantom.processors.action_processor import ActionProcessor
    from phantom.processors.smoothing_processor import SmoothingProcessor
    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor
    from phantom.processors.handinpaint_processor import HandInpaintProcessor
    from phantom.processors.synthetic_processor import SyntheticPickPlaceProcessor
    
    return {
        "bbox": BBoxProcessor,
        "hand2d": Hand2DProcessor,
        "hand3d": Hand3DProcessor,
        "hand_segmentation": HandSegmentationProcessor,
        "arm_segmentation": ArmSegmentationProcessor,
        "action": ActionProcessor,
        "smoothing": SmoothingProcessor,
        "robot_inpaint": RobotInpaintProcessor,
        "hand_inpaint": HandInpaintProcessor,
        SYNTHETIC_MODE: SyntheticPickPlaceProcessor,
    }

def validate_mode(cfg: DictConfig) -> None:
    """
    Validate that the mode parameter contains only valid processing modes.
    
    Args:
        cfg: Configuration object containing mode parameter
        
    Raises:
        ValueError: If mode contains invalid options
    """
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            modes = [mode.strip() for mode in cfg.mode.split(',')]
        else:
            modes = [cfg.mode]
    else:
        modes = cfg.mode
    
    # Get valid modes from enum
    valid_modes = {mode.value for mode in ProcessingMode}
    invalid_modes = [mode for mode in modes if mode not in valid_modes]
    
    if invalid_modes:
        valid_mode_list = [mode.value for mode in ProcessingMode]
        raise ValueError(
            f"Invalid mode(s): {invalid_modes}. "
            f"Valid modes are: {valid_mode_list}"
        )

def main(cfg: DictConfig):
    # Validate mode parameter
    validate_mode(cfg)
    
    # Get processor classes
    processor_classes = get_processor_classes(cfg)
    
    # Synthetic generation has no demo folders to iterate: run it once, then any per-demo modes
    if select_modes(cfg, synthetic=True):
        process_synthetic(cfg, processor_classes)
        if not select_modes(cfg):
            return

    if cfg.n_processes > 1:
        process_all_demos_parallel(cfg, processor_classes)
    elif cfg.demo_num is not None:
        process_one_demo(cfg.demo_num, cfg, processor_classes)
    else:
        process_all_demos(cfg, processor_classes)

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig):
    """
    Main entry point using Hydra configuration.
    
    Example usage:
    - Process all demos with bbox: python process_data.py mode=bbox
    - Process single demo: python process_data.py mode=bbox demo_num=0
    - Use EPIC dataset: python process_data.py dataset=epic mode=bbox
    - Parallel processing: python process_data.py mode=bbox n_processes=4
    - Process multiple modes sequentially: python process_data.py mode=bbox,hand3d
    - Process with custom order: python process_data.py mode=hand3d,bbox,action
    - Process with bracket notation (use quotes): python process_data.py "mode=[bbox,hand3d]"
    - Generate synthetic pick-and-place episodes: python process_data.py --config-name=ur7e mode=synthetic_pickplace
    """
    main(cfg)

if __name__ == "__main__":
    hydra_main()
