import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.application.session import RetargetingSession
from dex_mujoco.domain.models import HandFrame, RetargetingStepResult, SourceFrame


class _FakeEngine:
    def process(self, frame: HandFrame) -> RetargetingStepResult:
        return RetargetingStepResult(
            qpos=np.array([1.0, 2.0], dtype=np.float64),
            target_directions=None,
            processed_landmarks=frame.retarget_landmarks.copy(),
            handedness=frame.handedness,
        )


class _FakeSource:
    def __init__(self, frames):
        self.source_desc = "fake://source"
        self._frames = list(frames)
        self._index = 0
        self.closed = False

    @property
    def fps(self) -> int:
        return 30

    def is_available(self) -> bool:
        return self._index < len(self._frames)

    def get_frame(self) -> SourceFrame:
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    def stats_snapshot(self):
        return {}


class _FakeSink:
    def __init__(self):
        self.results = []
        self.closed = False

    @property
    def is_running(self) -> bool:
        return True

    def on_result(self, result: RetargetingStepResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


class _FakePreview:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def show(self, source, frame: SourceFrame) -> bool:
        self.calls += 1
        return True

    def close(self) -> None:
        self.closed = True


def _detection(handedness: str = "Right") -> HandFrame:
    landmarks = np.zeros((21, 3), dtype=np.float64)
    return HandFrame(
        landmarks_3d=landmarks,
        landmarks_2d=None,
        handedness=handedness,
    )


def test_session_runs_source_engine_and_sinks():
    source = _FakeSource(
        [
            SourceFrame(detection=_detection("Right")),
            SourceFrame(detection=None),
            SourceFrame(detection=_detection("Left")),
        ]
    )
    sink = _FakeSink()
    preview = _FakePreview()
    session = RetargetingSession(_FakeEngine(), sinks=[sink], preview_window=preview)

    summary = session.run(source, input_type="test")

    assert summary.source_desc == "fake://source"
    assert summary.input_type == "test"
    assert summary.num_frames == 3
    assert summary.num_detected == 2
    assert len(sink.results) == 2
    assert sink.results[1].handedness == "Left"
    assert source.closed is True
    assert sink.closed is True
    assert preview.closed is True
    assert preview.calls == 3
