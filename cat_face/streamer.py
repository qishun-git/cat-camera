from __future__ import annotations

import logging
import threading
import time
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
        self._target = str(target)
        self._fmt = str(fmt)
        self._options = {str(k): str(v) for k, v in (options or {}).items()}
        self._bitrate = max(int(bitrate), 1_000_000)
        self._lock = threading.Lock()
        self._active = False
        self._encoder = None
        self._output = None
        self._restart_thread: Optional[threading.Thread] = None
        self._start_locked()

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            self._start_locked()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    # Internal helpers -------------------------------------------------

    def _create_output(self):
        output = PyavOutput(self._target, format=self._fmt, options=self._options)  # type: ignore[name-defined]
        if hasattr(output, "error_callback"):
            output.error_callback = self._handle_output_error  # type: ignore[attr-defined]
        return output

    def _start_locked(self) -> None:
        self._encoder = H264Encoder(bitrate=self._bitrate)  # type: ignore[name-defined]
        self._output = self._create_output()
        self._picamera.start_encoder(self._encoder, self._output)  # type: ignore[attr-defined]
        self._active = True
        logger.info("Streaming publisher connected to %s (%s)", self._target, self._fmt)

    def _stop_locked(self) -> None:
        if self._encoder is None:
            return
        try:
            if self._active:
                self._picamera.stop_encoder(self._encoder)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to stop streaming encoder cleanly: %s", exc)
        finally:
            self._active = False
            self._encoder = None
            self._output = None

    def _handle_output_error(self, exc: Exception) -> None:
        logger.warning("Streaming output error: %s", exc)
        def restart():
            with self._lock:
                self._stop_locked()
                time.sleep(0.5)
                try:
                    self._start_locked()
                except Exception as restart_exc:
                    logger.error("Unable to restart streaming publisher: %s", restart_exc)
        thread = threading.Thread(target=restart, daemon=True)
        thread.start()
        self._restart_thread = thread
