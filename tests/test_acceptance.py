import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.acceptance import mirror_pose_to_left, rotation_matrix, synthetic_hand_pose
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import compute_target_directions


def test_config_resolves_absolute_mjcf_path():
    config = RetargetingConfig.load("configs/retargeting/linkerhand_l20.yaml")
    assert Path(config.hand.mjcf_path).is_absolute()
    assert Path(config.hand.mjcf_path).exists()


def test_wrist_local_preprocess_is_rotation_invariant():
    config = RetargetingConfig.load("configs/retargeting/linkerhand_l20.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    base_pose = synthetic_hand_pose("open")
    base_dirs = compute_target_directions(
        base_pose,
        vector_pairs,
        handedness="Right",
        frame=config.preprocess.frame,
    )
    rotated_pose = base_pose @ rotation_matrix("z", 70.0).T
    rotated_dirs = compute_target_directions(
        rotated_pose,
        vector_pairs,
        handedness="Right",
        frame=config.preprocess.frame,
    )
    cosine = float(np.mean(np.sum(base_dirs * rotated_dirs, axis=1)))
    assert cosine > 0.98


def test_left_and_right_inputs_match_after_mirroring():
    config = RetargetingConfig.load("configs/retargeting/linkerhand_l20.yaml")
    vector_pairs = [(a, b) for a, b in config.human_vector_pairs]
    right_pose = synthetic_hand_pose("pinch")
    left_pose = mirror_pose_to_left(right_pose)
    right_dirs = compute_target_directions(
        right_pose,
        vector_pairs,
        handedness="Right",
        frame=config.preprocess.frame,
    )
    left_dirs = compute_target_directions(
        left_pose,
        vector_pairs,
        handedness="Left",
        frame=config.preprocess.frame,
    )
    cosine = float(np.mean(np.sum(right_dirs * left_dirs, axis=1)))
    assert cosine > 0.98
