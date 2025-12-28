from __future__ import annotations

import threading
import time
from http import server
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import json


class MJPEGStreamer:
    """Simple MJPEG streaming server fed with existing frames."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        resolution: Optional[Tuple[int, int]] = None,
        quality: int = 80,
        frame_interval: float = 0.03,
        status_path: Optional[str] = None,
    ) -> None:
        self._resolution = resolution
        self._quality = max(10, min(int(quality), 95))
        self._condition = threading.Condition()
        self._frame: Optional[bytes] = None
        self._shutdown = False
        self._frame_interval = max(0.0, float(frame_interval))
        self._clients = 0
        self._status_path = status_path

        handler = self._build_handler()
        self._server = server.ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._write_status_locked()
        print(f"MJPEG stream available at http://{host}:{port}/stream.mjpg")

    def _build_handler(self):
        streamer = self

        class StreamHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                request_path = self.path.split("?")[0]
                if request_path not in ("/", "/stream", "/stream.mjpg"):
                    self.send_error(404)
                    return
                is_stream = request_path == "/stream.mjpg"
                if is_stream:
                    streamer._increment_clients()
                if is_stream:
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
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        streamer._decrement_clients()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Pi-Cam streamer")

            def log_message(self, format, *args):
                return

        return StreamHandler

    def wait_for_frame(self) -> Optional[bytes]:
        with self._condition:
            while self._frame is None and not self._shutdown:
                self._condition.wait()
            return self._frame

    def push_frame(self, frame: np.ndarray) -> None:
        if frame is None or self._clients == 0:
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

    def _increment_clients(self) -> None:
        with self._condition:
            self._clients += 1
            self._write_status_locked()

    def _decrement_clients(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)
            self._write_status_locked()

    @property
    def client_count(self) -> int:
        with self._condition:
            return self._clients

    def _write_status_locked(self) -> None:
        if not self._status_path:
            return
        path = Path(self._status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"clients": self._clients}))

    def stop(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
            self._clients = 0
            self._write_status_locked()
        self._server.shutdown()
        self._server.server_close()
