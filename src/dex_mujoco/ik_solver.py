"""IK solver configuration and utilities."""

from dataclasses import dataclass


@dataclass
class SolveParams:
    """Parameters for the vector retargeting solver."""
    step_size: float = 0.15
    max_iterations: int = 30
