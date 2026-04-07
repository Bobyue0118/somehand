"""Serialization of runtime artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def save_trajectory_artifact(
    output_path: str | None,
    trajectory: list[np.ndarray],
    *,
    joint_names: list[str],
    config_path: str,
    num_frames: int,
    source_desc: str,
    input_type: str,
    handedness: str | None = None,
    num_detected: int | None = None,
) -> None:
    if not output_path or not trajectory:
        return

    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trajectory": np.array(trajectory),
        "joint_names": joint_names,
        "config_path": config_path,
        "num_frames": num_frames,
        "input_source": source_desc,
        "input_type": input_type,
    }
    if handedness is not None:
        payload["handedness"] = handedness
    if num_detected is not None:
        payload["num_detected"] = num_detected

    with artifact_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    print(f"Saved trajectory ({len(trajectory)} frames) to {artifact_path}")
