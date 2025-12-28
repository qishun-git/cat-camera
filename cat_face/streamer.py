from __future__ import annotations

import threading
import time
from http import server
from typing import Optional, Tuple

import cv2
import numpy as np


class MJPEGStreamer:
    """Simple MJPEG streaming server fed with existing frames."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        resolution: Optional[Tuple[int, int]] = None,
        quality: int = 80,
        frame_interval: float = 0.03,
    ) -> None:
        self._resolution = resolution
        self._quality = max(10, min(int(quality), 95))
        self._condition = threading.Condition()
        self._frame: Optional[bytes] = None
        self._shutdown = False
        self._frame_interval = max(0.0, float(frame_interval))

        handler = self._build_handler()
        self._server = server.ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"MJPEG stream available at http://{host}:{port}/stream.mjpg")

    def _build_handler(self):
        streamer = self

        class StreamHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                request_path = self.path.split("?")[0]
                if request_path not in ("/", "/stream", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = streamer.wait_for_frame()
                        if frame is None:
                            break
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        if streamer._frame_interval:
                            time.sleep(streamer._frame_interval)
                except BrokenPipeError:
                    pass

            def log_message(self, format, *args):
                return

        return StreamHandler

    def wait_for_frame(self) -> Optional[bytes]:
        with self._condition:
            while self._frame is None and not self._shutdown:
                self._condition.wait()
            return self._frame

    def push_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        display = frame
        if self._resolution:
            display = cv2.resize(display, self._resolution, interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not success:
            return
        with self._condition:
            self._frame = buffer.tobytes()
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
