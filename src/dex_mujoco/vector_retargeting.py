"""Core vector retargeting algorithm.

Uses weighted cosine similarity loss (frame-invariant direction matching),
analytical MuJoCo Jacobians, and scipy SLSQP optimizer.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize

from .hand_model import HandModel
from .retargeting_config import RetargetingConfig

_OPERATOR2ROBOT_RIGHT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _mediapipe_to_mujoco(landmarks_3d: np.ndarray) -> np.ndarray:
    """Transform MediaPipe landmarks to robot-aligned frame."""
    out = np.empty_like(landmarks_3d)
    out[:, 0] = -landmarks_3d[:, 2]
    out[:, 1] = landmarks_3d[:, 0]
    out[:, 2] = -landmarks_3d[:, 1]
    return out


def _mirror_left_to_right(landmarks_3d: np.ndarray) -> np.ndarray:
    mirrored = landmarks_3d.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def _estimate_wrist_frame(landmarks_3d: np.ndarray) -> np.ndarray:
    palm = landmarks_3d[[0, 5, 9, 13, 17], :]
    x_vector = landmarks_3d[0] - landmarks_3d[9]
    palm_centered = palm - palm.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(palm_centered, full_matrices=False)
    normal = vh[-1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        raise ValueError("Cannot estimate palm normal from degenerate landmarks")
    normal = normal / normal_norm

    x_axis = x_vector - np.dot(x_vector, normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Cannot estimate palm x-axis from degenerate landmarks")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, normal)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Cannot estimate palm z-axis from degenerate landmarks")
    z_axis = z_axis / z_norm

    if np.dot(z_axis, landmarks_3d[17] - landmarks_3d[5]) < 0.0:
        normal *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, normal, z_axis], axis=1)


def preprocess_landmarks(
    landmarks_3d: np.ndarray,
    handedness: str = "Right",
    frame: str = "wrist_local",
) -> np.ndarray:
    if handedness == "Left":
        landmarks_3d = _mirror_left_to_right(landmarks_3d)

    centered = landmarks_3d - landmarks_3d[0:1, :]
    if frame == "camera_aligned":
        return _mediapipe_to_mujoco(centered)
    if frame != "wrist_local":
        raise ValueError(f"Unsupported preprocess frame: {frame}")

    try:
        wrist_frame = _estimate_wrist_frame(centered)
        return centered @ wrist_frame @ _OPERATOR2ROBOT_RIGHT
    except ValueError:
        return _mediapipe_to_mujoco(centered)


def compute_target_directions(
    landmarks_3d: np.ndarray,
    human_vector_pairs: list[tuple[int, int]],
    handedness: str = "Right",
    frame: str = "wrist_local",
) -> np.ndarray:
    landmarks = preprocess_landmarks(landmarks_3d, handedness=handedness, frame=frame)
    directions = np.empty((len(human_vector_pairs), 3), dtype=np.float64)
    for i, (origin_idx, target_idx) in enumerate(human_vector_pairs):
        vector = landmarks[target_idx] - landmarks[origin_idx]
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            directions[i] = 0.0
        else:
            directions[i] = vector / norm
    return directions


class TemporalFilter:
    """Exponential moving average filter for smooth landmark tracking."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def filter(self, value: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value.copy()
            return value
        self._prev = self.alpha * value + (1 - self.alpha) * self._prev
        return self._prev.copy()

    def reset(self):
        self._prev = None


class VectorRetargeter:
    """Optimizes robot joint angles to match human finger vector directions."""

    def __init__(self, hand_model: HandModel, config: RetargetingConfig):
        self.hand_model = hand_model
        self.config = config
        self.model = hand_model.model
        self.data = hand_model.data

        self.human_vector_pairs = [
            (pair[0], pair[1]) for pair in config.human_vector_pairs
        ]
        self.origin_link_names = config.origin_link_names
        self.task_link_names = config.task_link_names

        self.origin_ids = []
        self.origin_is_site = []
        for i, name in enumerate(self.origin_link_names):
            is_site = config.origin_link_types[i] == "site"
            self.origin_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            link_id = mujoco.mj_name2id(self.model, obj_type, name)
            assert link_id >= 0, f"Origin link '{name}' not found in model"
            self.origin_ids.append(link_id)

        self.task_ids = []
        self.task_is_site = []
        for i, name in enumerate(self.task_link_names):
            is_site = config.task_link_types[i] == "site"
            self.task_is_site.append(is_site)
            obj_type = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
            link_id = mujoco.mj_name2id(self.model, obj_type, name)
            assert link_id >= 0, f"Task link '{name}' not found in model"
            self.task_ids.append(link_id)

        self.landmark_filter = TemporalFilter(
            alpha=config.preprocess.temporal_filter_alpha
        )

        self._norm_delta = config.solver.norm_delta
        self._max_iterations = config.solver.max_iterations
        self._output_alpha = config.solver.output_alpha
        self._weights = np.array(config.vector_weights, dtype=np.float64)
        self._preprocess_frame = config.preprocess.frame

        self._target_directions: np.ndarray | None = None
        self._last_qpos: np.ndarray | None = None

        nq = self.model.nq
        self._bounds = []
        for j in range(nq):
            lo, hi = self.model.jnt_range[j]
            if lo < hi:
                self._bounds.append((float(lo), float(hi)))
            else:
                self._bounds.append((None, None))

    def _forward(self, qpos: np.ndarray | None = None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.data)

    def _get_pos(self, idx: int, is_site: bool) -> np.ndarray:
        if is_site:
            return self.data.site_xpos[idx].copy()
        return self.data.xpos[idx].copy()

    def _get_robot_vectors(self) -> np.ndarray:
        vectors = np.empty((len(self.origin_ids), 3))
        for i in range(len(self.origin_ids)):
            p_origin = self._get_pos(self.origin_ids[i], self.origin_is_site[i])
            p_task = self._get_pos(self.task_ids[i], self.task_is_site[i])
            vectors[i] = p_task - p_origin
        return vectors

    def _compute_loss(self, qpos: np.ndarray) -> float:
        self._forward(qpos)
        robot_vecs = self._get_robot_vectors()
        loss = 0.0
        for i in range(len(robot_vecs)):
            r_norm = np.linalg.norm(robot_vecs[i])
            if r_norm < 1e-8:
                loss += self._weights[i]
                continue
            cos_sim = np.dot(robot_vecs[i] / r_norm, self._target_directions[i])
            loss += self._weights[i] * (1.0 - cos_sim)
        if self._last_qpos is not None:
            loss += self._norm_delta * np.sum((qpos - self._last_qpos) ** 2)
        return loss

    def _compute_loss_and_grad(self, qpos: np.ndarray) -> tuple[float, np.ndarray]:
        self._forward(qpos)
        robot_vecs = self._get_robot_vectors()
        nv = self.model.nv
        grad = np.zeros(nv)
        loss = 0.0

        for i in range(len(self.origin_ids)):
            r_vec = robot_vecs[i]
            r_norm = np.linalg.norm(r_vec)
            if r_norm < 1e-8:
                loss += self._weights[i]
                continue

            r_dir = r_vec / r_norm
            t_dir = self._target_directions[i]
            cos_sim = np.dot(r_dir, t_dir)
            loss += self._weights[i] * (1.0 - cos_sim)

            grad_vec = -(t_dir - cos_sim * r_dir) / r_norm
            jac_task = np.zeros((3, nv))
            jac_origin = np.zeros((3, nv))

            if self.task_is_site[i]:
                mujoco.mj_jacSite(self.model, self.data, jac_task, None, self.task_ids[i])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_task, None, self.task_ids[i])

            if self.origin_is_site[i]:
                mujoco.mj_jacSite(self.model, self.data, jac_origin, None, self.origin_ids[i])
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_origin, None, self.origin_ids[i])

            grad += self._weights[i] * (grad_vec @ (jac_task - jac_origin))

        if self._last_qpos is not None:
            delta_q = qpos - self._last_qpos
            loss += self._norm_delta * np.sum(delta_q ** 2)
            grad += 2.0 * self._norm_delta * delta_q

        return loss, grad

    def update_targets(self, landmarks_3d: np.ndarray, handedness: str = "Right"):
        landmarks = preprocess_landmarks(
            landmarks_3d,
            handedness=handedness,
            frame=self._preprocess_frame,
        )
        landmarks = self.landmark_filter.filter(landmarks)

        directions = np.empty((len(self.human_vector_pairs), 3), dtype=np.float64)
        for i, (origin_idx, target_idx) in enumerate(self.human_vector_pairs):
            v = landmarks[target_idx] - landmarks[origin_idx]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                directions[i] = 0.0
            else:
                directions[i] = v / norm
        self._target_directions = directions

    def solve(self) -> np.ndarray:
        if self._target_directions is None:
            return self.data.qpos.copy()

        x0 = self.data.qpos.copy()
        previous_qpos = None if self._last_qpos is None else self._last_qpos.copy()

        result = minimize(
            fun=self._compute_loss_and_grad,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=self._bounds,
            options={
                "maxiter": self._max_iterations,
                "ftol": 1e-6,
            },
        )

        qpos = result.x.copy()
        if previous_qpos is not None and self._output_alpha < 1.0:
            qpos = previous_qpos + self._output_alpha * (qpos - previous_qpos)

        self._last_qpos = qpos.copy()
        self._forward(qpos)
        return qpos

    def compute_error(self) -> float:
        self._forward()
        if self._target_directions is None:
            return 0.0
        return self._compute_loss(self.data.qpos.copy())

    def get_target_directions(self) -> np.ndarray | None:
        if self._target_directions is None:
            return None
        return self._target_directions.copy()
