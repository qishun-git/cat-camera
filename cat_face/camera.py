from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2

    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore
    PICAMERA2_AVAILABLE = False


class CameraError(RuntimeError):
    """Raised when a camera cannot be initialized."""


class CameraInterface:
    """Common interface for camera backends."""

    def read(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def is_open(self) -> bool:
        return True


class OpenCVCamera(CameraInterface):
    def __init__(self, camera_index: int, resolution: Optional[Tuple[int, int]] = None) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise CameraError(f"Unable to open camera index {camera_index}")
        if resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self) -> Tuple[bool, np.ndarray]:
        return self._cap.read()

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_open(self) -> bool:
        return self._cap.isOpened()


class Picamera2Camera(CameraInterface):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
    ) -> None:
        if not PICAMERA2_AVAILABLE:
            raise CameraError("Picamera2 is not available on this system.")
        self._picam = Picamera2()
        main_config = {"format": "RGB888"}
        if resolution:
            main_config["size"] = resolution
        self._fps = float(target_fps) if target_fps and target_fps > 0 else 0.0
        controls: Dict[str, Any] = {}
        if self._fps > 0:
            frame_duration = int(1_000_000 / self._fps)
            controls["FrameDurationLimits"] = (frame_duration, frame_duration)
        config = self._picam.create_video_configuration(main=main_config, controls=controls)
        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.05)
        if self._fps <= 0:
            self._fps = self._measure_running_fps()
        if self._fps <= 0:
            self._fps = 30.0

    def read(self) -> Tuple[bool, np.ndarray]:
        frame = self._picam.capture_array("main")
        if frame is None:
            return False, np.zeros((1, 1, 3), dtype=np.uint8)
        converted = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, converted

    def release(self) -> None:
        if self._picam:
            self._picam.close()

    def _measure_running_fps(self) -> float:
        for _ in range(5):
            try:
                metadata = self._picam.capture_metadata()
            except Exception:
                metadata = None
            frame_duration = None
            if isinstance(metadata, dict):
                frame_duration = metadata.get("FrameDuration")
                if not frame_duration:
                    limits = metadata.get("FrameDurationLimits")
                    if isinstance(limits, (list, tuple)) and limits:
                        frame_duration = limits[0]
            if frame_duration:
                try:
                    fps = 1_000_000.0 / float(frame_duration)
                    if fps > 0:
                        return fps
                except (TypeError, ZeroDivisionError):
                    pass
            time.sleep(0.02)
        return 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_open(self) -> bool:
        return True

    @property
    def raw(self):
        return self._picam


def create_camera(
    camera_index: int = 0,
    prefer_picamera: bool = False,
    picamera_resolution: Optional[Tuple[int, int]] = None,
    picamera_fps: Optional[float] = None,
    opencv_resolution: Optional[Tuple[int, int]] = None,
) -> CameraInterface:
    if prefer_picamera:
        if not PICAMERA2_AVAILABLE:
            raise CameraError(
                "Picamera2 was requested but is not available. Install picamera2 or disable 'prefer_picamera2'."
            )
        return Picamera2Camera(resolution=picamera_resolution, target_fps=picamera_fps)
    return OpenCVCamera(camera_index, resolution=opencv_resolution)
