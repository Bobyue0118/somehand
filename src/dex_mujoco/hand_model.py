"""MuJoCo hand model abstraction with Mink configuration."""

from pathlib import Path

import mujoco
import numpy as np

import mink


class HandModel:
    """Wraps a MuJoCo hand model and provides kinematic queries via Mink."""

    def __init__(self, mjcf_path: str):
        """Load a hand model from MJCF file.

        Args:
            mjcf_path: Path to the MJCF XML file.
        """
        self.mjcf_path = str(Path(mjcf_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get world position of a body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_site_position(self, site_name: str) -> np.ndarray:
        """Get world position of a site."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()

    def get_joint_names(self) -> list[str]:
        """Get names of all joints."""
        names = []
        for i in range(self.model.njnt):
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
        return names

    def get_body_names(self) -> list[str]:
        """Get names of all bodies (excluding world body 0)."""
        names = []
        for i in range(1, self.model.nbody):
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i))
        return names

    def get_site_names(self) -> list[str]:
        """Get names of all sites."""
        names = []
        for i in range(self.model.nsite):
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i))
        return names

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions."""
        return self.data.qpos.copy()

    def set_qpos(self, qpos: np.ndarray):
        """Set joint positions and run forward kinematics."""
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset to default configuration."""
        mujoco.mj_resetData(self.model, self.data)
        self.configuration.update()
