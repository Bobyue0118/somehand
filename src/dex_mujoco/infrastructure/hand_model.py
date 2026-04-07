"""MuJoCo hand-model adapter."""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np


class HandModel:
    """Wraps a MuJoCo hand model and provides kinematic queries via Mink."""

    def __init__(self, mjcf_path: str):
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
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_site_position(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()

    def get_joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(self.model.njnt)
        ]

    def get_body_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(1, self.model.nbody)
        ]

    def get_site_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            for site_id in range(self.model.nsite)
        ]

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.configuration.update()
