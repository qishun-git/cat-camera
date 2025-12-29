from __future__ import annotations

import logging
from typing import Dict, Optional

from cat_face.camera import PICAMERA2_AVAILABLE

logger = logging.getLogger(__name__)

try:
    if PICAMERA2_AVAILABLE:
        from picamera2 import Picamera2  # type: ignore
        from picamera2.encoders import H264Encoder  # type: ignore
        from picamera2.outputs import PyavOutput  # type: ignore
    else:
        Picamera2 = None  # type: ignore
        H264Encoder = None  # type: ignore
        PyavOutput = None  # type: ignore
except ImportError:  # pragma: no cover - picamera2 not available on macOS dev machines
    Picamera2 = None  # type: ignore
    H264Encoder = None  # type: ignore
    PyavOutput = None  # type: ignore


class PicameraRTSPPublisher:
    """Publishes the Picamera2 main stream to an RTSP/RTMP/HLS endpoint via PyAV."""

    def __init__(
        self,
        picamera: Picamera2,  # type: ignore[valid-type]
        target: str,
        bitrate: int,
        fmt: str = "rtsp",
        options: Optional[Dict[str, str]] = None,
    ) -> None:
        if Picamera2 is None or H264Encoder is None or PyavOutput is None:
            raise RuntimeError("Streaming requested but Picamera2/PyAV components are unavailable.")
        if not target:
            raise ValueError("Streaming target URL must be provided.")
        self._picamera = picamera
        self._encoder = H264Encoder(bitrate=max(int(bitrate), 1_000_000))  # type: ignore[name-defined]
        opts = {str(k): str(v) for k, v in (options or {}).items()}
        self._output = PyavOutput(target, format=fmt, options=opts)  # type: ignore[name-defined]
        self._active = False
        self.start()

    def start(self) -> None:
        if self._active:
            return
        self._picamera.start_encoder(self._encoder, self._output)  # type: ignore[attr-defined]
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        try:
            self._picamera.stop_encoder(self._encoder)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to stop streaming encoder cleanly: %s", exc)
        finally:
            self._active = False
