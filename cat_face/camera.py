from __future__ import annotations

from typing import Tuple

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
