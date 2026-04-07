"""Session-level orchestration for sources, engine, preview, and sinks."""

from __future__ import annotations

import time
from collections.abc import Sequence

from dex_mujoco.domain import HandTrackingSource, OutputSink, PreviewWindow, SessionSummary

from .engine import RetargetingEngine


class RetargetingSession:
    """Runs the main input -> engine -> sink loop."""

    def __init__(
        self,
        engine: RetargetingEngine,
        *,
        sinks: Sequence[OutputSink] = (),
        preview_window: PreviewWindow | None = None,
    ):
        self.engine = engine
        self.sinks = list(sinks)
        self.preview_window = preview_window

    @property
    def is_running(self) -> bool:
        return all(sink.is_running for sink in self.sinks)

    def run(
        self,
        source: HandTrackingSource,
        *,
        input_type: str,
        realtime: bool = False,
        loop: bool = False,
        stats_every: int = 0,
    ) -> SessionSummary:
        frame_count = 0
        detected_count = 0
        frame_period = 1.0 / max(source.fps, 1)

        try:
            while True:
                if not source.is_available():
                    if loop and source.reset():
                        continue
                    break

                tic = time.monotonic()
                try:
                    frame = source.get_frame()
                except StopIteration:
                    break

                frame_count += 1

                if frame.detection is not None:
                    detected_count += 1
                    result = self.engine.process(frame.detection)
                    for sink in self.sinks:
                        sink.on_result(result)

                if self.preview_window is not None and not self.preview_window.show(source, frame):
                    break

                if stats_every > 0 and frame_count % stats_every == 0:
                    stats = source.stats_snapshot()
                    if stats:
                        print(
                            "UDP stats:"
                            f" recv={stats.get('packets_received', 0)}"
                            f" valid={stats.get('packets_valid', 0)}"
                            f" bad_size={stats.get('packets_bad_size', 0)}"
                            f" bad_decode={stats.get('packets_bad_decode', 0)}"
                            f" floats={stats.get('last_float_count', 0)}/{stats.get('expected_float_count', 0)}"
                            f" bytes={stats.get('last_packet_bytes', 0)}"
                            f" sender={stats.get('last_sender')}"
                        )

                if not self.is_running:
                    break

                if realtime:
                    elapsed = time.monotonic() - tic
                    sleep_s = frame_period - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            source.close()
            if self.preview_window is not None:
                self.preview_window.close()
            for sink in reversed(self.sinks):
                sink.close()

        return SessionSummary(
            num_frames=frame_count,
            num_detected=detected_count,
            source_desc=source.source_desc,
            input_type=input_type,
        )
