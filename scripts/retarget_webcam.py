"""Real-time hand retargeting from webcam."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.hand_detector import HandDetector
from dex_mujoco.hand_model import HandModel
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import VectorRetargeter
from dex_mujoco.visualization import HandVisualizer


def main():
    parser = argparse.ArgumentParser(description="Real-time hand retargeting from webcam")
    parser.add_argument(
        "--config",
        default="configs/retargeting/linkerhand_l20.yaml",
        help="Path to retargeting config YAML",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--hand",
        choices=["Left", "Right"],
        default="Right",
        help="Actual operator hand to retarget",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if your capture pipeline reports the opposite hand",
    )
    args = parser.parse_args()

    config = RetargetingConfig.load(args.config)
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)
    detector = HandDetector(target_hand=args.hand, swap_handedness=args.swap_hands)
    visualizer = HandVisualizer(hand_model)

    print(f"Model: {config.hand.name} ({hand_model.nq} DOF)")
    print(f"Retargeting: {len(config.human_vector_pairs)} vector pairs")
    print(f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}")
    print("Press 'q' to quit.")

    for frame in HandDetector.create_source(args.camera):
        detection = detector.detect(frame)

        if detection is not None:
            retargeter.update_targets(detection.landmarks_3d, detection.handedness)
            qpos = retargeter.solve()
            visualizer.update(qpos)
            frame = detector.draw_landmarks(frame, detection)

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if not visualizer.is_running:
            break

    detector.close()
    visualizer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
