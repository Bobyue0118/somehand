"""OpenCV preview window adapter."""

from __future__ import annotations

import cv2

from dex_mujoco.domain import SourceFrame


class OpenCvPreviewWindow:
    def __init__(self, window_name: str = "Hand Detection"):
        self.window_name = window_name

    def show(self, source: object, frame: SourceFrame) -> bool:
        if frame.preview_frame is None:
            return True

        preview = frame.preview_frame
        annotate_preview = getattr(source, "annotate_preview", None)
        if frame.detection is not None and callable(annotate_preview):
            preview = annotate_preview(preview, frame.detection)

        cv2.imshow(self.window_name, preview)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()
