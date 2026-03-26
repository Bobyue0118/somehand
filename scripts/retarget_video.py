"""Offline hand retargeting from video file."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.hand_detector import HandDetector
from dex_mujoco.hand_model import HandModel
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import VectorRetargeter
from dex_mujoco.visualization import HandVisualizer


def main():
    parser = argparse.ArgumentParser(description="Offline hand retargeting from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--config",
        default="configs/retargeting/linkerhand_l20.yaml",
        help="Path to retargeting config YAML",
    )
    parser.add_argument("--output", default=None, help="Output pickle file for joint trajectory")
    parser.add_argument("--visualize", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument(
        "--hand",
        choices=["Left", "Right"],
        default="Right",
        help="Actual operator hand to retarget",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if this video reports the opposite hand",
    )
    args = parser.parse_args()

    config = RetargetingConfig.load(args.config)
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)
    detector = HandDetector(target_hand=args.hand, swap_handedness=args.swap_hands)

    visualizer = None
    if args.visualize:
        visualizer = HandVisualizer(hand_model)

    trajectory = []
    frame_count = 0
    detected_count = 0

    print(f"Processing video: {args.video}")

    for frame in HandDetector.create_source(args.video):
        frame_count += 1
        detection = detector.detect(frame)

        if detection is not None:
            detected_count += 1
            retargeter.update_targets(detection.landmarks_3d, detection.handedness)
            qpos = retargeter.solve()
            trajectory.append(qpos)

            if visualizer is not None:
                visualizer.update(qpos)
                if not visualizer.is_running:
                    break

    print(f"Processed {frame_count} frames, detected hand in {detected_count} frames")

    if args.output and trajectory:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "trajectory": np.array(trajectory),
            "joint_names": hand_model.get_joint_names(),
            "config_path": args.config,
            "num_frames": frame_count,
            "num_detected": detected_count,
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved trajectory ({len(trajectory)} frames) to {args.output}")

    detector.close()
    if visualizer is not None:
        visualizer.close()


if __name__ == "__main__":
    main()
