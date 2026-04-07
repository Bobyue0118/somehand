"""Infrastructure adapters for external systems."""

from .artifacts import save_trajectory_artifact
from .config_loader import load_retargeting_config
from .hand_model import HandModel
from .preview import OpenCvPreviewWindow
from .sinks import AsyncLandmarkOutputSink, RobotHandOutputSink, TrajectoryRecorder
from .sources import HCMocapInputSource, MediaPipeInputSource, create_hc_mocap_bvh_source, create_hc_mocap_udp_source, create_pico_source
from .vector_solver import VectorRetargeter

__all__ = [
    "AsyncLandmarkOutputSink",
    "HCMocapInputSource",
    "HandModel",
    "MediaPipeInputSource",
    "OpenCvPreviewWindow",
    "RobotHandOutputSink",
    "TrajectoryRecorder",
    "VectorRetargeter",
    "create_hc_mocap_bvh_source",
    "create_hc_mocap_udp_source",
    "create_pico_source",
    "load_retargeting_config",
    "save_trajectory_artifact",
]
