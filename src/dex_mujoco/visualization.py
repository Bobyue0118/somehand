"""MuJoCo passive viewer for real-time hand visualization."""

import mujoco
import mujoco.viewer
import numpy as np

from .hand_model import HandModel


class HandVisualizer:
    """Real-time MuJoCo hand visualization using passive viewer."""

    def __init__(self, hand_model: HandModel):
        self.hand_model = hand_model
        self.model = hand_model.model
        self.data = hand_model.data
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )

    def update(self, qpos: np.ndarray):
        """Update visualization with new joint positions."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
