from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

import cv2
import numpy as np

from cat_face.camera import CameraError, CameraInterface, PICAMERA2_AVAILABLE
from cat_face.utils import configure_logging, ensure_dir, load_project_config, resolve_paths

logger = logging.getLogger(__name__)

if PICAMERA2_AVAILABLE:
    from picamera2 import Picamera2  # type: ignore
    from picamera2.encoders import H264Encoder  # type: ignore
    from picamera2.outputs import FfmpegOutput, PyavOutput  # type: ignore
else:
    Picamera2 = None  # type: ignore
    H264Encoder = None  # type: ignore
    FfmpegOutput = None  # type: ignore
    PyavOutput = None  # type: ignore


@dataclass
class RecorderConfig:
    output_dir: Path
    min_duration: float
    max_duration: float
    cooldown: float
    absence_grace: float
    fps_override: Optional[float]
    codec: str
    picamera_bitrate: int

    @classmethod
    def from_project(cls, project_config: Dict[str, Any], base_path: Path) -> "RecorderConfig":
        defaults = {
            "output_dir": str(base_path / "clips"),
            "min_duration": 15.0,
            "max_duration": 30.0,
            "cooldown": 5.0,
            "absence_grace": 1.0,
            "fps": 0.0,
            "codec": "mp4v",
            "picamera_bitrate": 10_000_000,
        }
        raw_cfg = defaults | (project_config.get("recorder") or {})
        min_duration = max(float(raw_cfg["min_duration"]), 0.0)
        max_duration = max(float(raw_cfg["max_duration"]), min_duration)
        cooldown = max(float(raw_cfg["cooldown"]), 0.0)
        absence_grace = max(float(raw_cfg["absence_grace"]), 0.0)
        fps_value = float(raw_cfg.get("fps", 0.0))
        fps_override = fps_value if fps_value > 0 else None
        output_dir = ensure_dir(Path(str(raw_cfg["output_dir"])))

        return cls(
            output_dir=output_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            cooldown=cooldown,
            absence_grace=absence_grace,
            fps_override=fps_override,
            codec=str(raw_cfg.get("codec", "mp4v")),
            picamera_bitrate=int(raw_cfg.get("picamera_bitrate", 10_000_000)),
        )


class RecordingBackend(ABC):
    @abstractmethod
    def start(self, frame: np.ndarray, temp_path: Path, fps: float) -> None:
        """Prepare backend resources for a new clip."""

    @abstractmethod
    def handle_frame(self, frame: np.ndarray) -> None:
        """Consume another frame (no-op for Picamera)."""

    @abstractmethod
    def stop(self) -> None:
        """Release backend resources (finalize or discard)."""


class OpenCVRecordingBackend(RecordingBackend):
    def __init__(self, fourcc: int) -> None:
        self._fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None

    def start(self, frame: np.ndarray, temp_path: Path, fps: float) -> None:
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(temp_path), self._fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Unable to create video writer for clip recording.")
        self._writer = writer

    def handle_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class PicameraRecordingBackend(RecordingBackend):
    def __init__(self, picamera: Picamera2, bitrate: int) -> None:  # type: ignore[valid-type]
        if Picamera2 is None or H264Encoder is None or FfmpegOutput is None:
            raise RuntimeError("Picamera2 native backend requested but Picamera2 is unavailable.")
        self._picamera = picamera
        self._bitrate = max(int(bitrate), 1_000_000)
        self._encoder: Optional[H264Encoder] = None  # type: ignore[name-defined]
        self._output: Optional[FfmpegOutput] = None  # type: ignore[name-defined]

    def start(self, frame: np.ndarray, temp_path: Path, fps: float) -> None:
        self._encoder = H264Encoder(bitrate=self._bitrate)  # type: ignore[name-defined]
        self._output = FfmpegOutput(str(temp_path))  # type: ignore[name-defined]
        self._picamera.start_encoder(self._encoder, self._output)  # type: ignore[attr-defined, call-arg]

    def handle_frame(self, frame: np.ndarray) -> None:
        # Picamera recordings are handled directly by the ISP.
        return

    def stop(self) -> None:
        if self._encoder is None:
            return
        try:
            self._picamera.stop_encoder()  # type: ignore[attr-defined, call-arg]
        finally:
            if self._output and hasattr(self._output, "close"):
                try:
                    self._output.close()  # type: ignore[call-arg]
                except Exception:
                    pass
            self._encoder = None
            self._output = None


class SharedPicameraEncoder:
    """Keeps a single Picamera2 encoder running for both streaming and clip recording."""

    def __init__(
        self,
        picamera: Picamera2,  # type: ignore[valid-type]
        bitrate: int,
        publish_url: str,
        publish_format: str,
        publish_options: Dict[str, str],
    ) -> None:
        if PyavOutput is None or H264Encoder is None or FfmpegOutput is None:
            raise RuntimeError("Picamera2 streaming requested but required modules are unavailable.")
        if not publish_url:
            raise ValueError("publish_url must be provided for streaming.")
        self._picamera = picamera
        self._encoder = H264Encoder(bitrate=max(int(bitrate), 1_000_000))  # type: ignore[name-defined]
        self._lock = threading.Lock()
        self._base_outputs: List[Any] = []
        self._clip_output: Optional[FfmpegOutput] = None  # type: ignore[name-defined]
        stream_output = PyavOutput(publish_url, format=publish_format, options=publish_options)  # type: ignore[name-defined]
        self._stream_output = stream_output
        self._base_outputs.append(stream_output)
        self._encoder.output = list(self._base_outputs)
        self._picamera.start_encoder(self._encoder)  # type: ignore[attr-defined]
        logger.info("Publishing live stream to %s (%s)", publish_url, publish_format)

    def start_clip(self, temp_path: Path) -> None:
        with self._lock:
            if self._clip_output is not None:
                raise RuntimeError("Recorder already writing a clip.")
            clip_output = FfmpegOutput(str(temp_path))  # type: ignore[name-defined]
            outputs = list(self._base_outputs)
            outputs.append(clip_output)
            self._encoder.output = outputs
            self._clip_output = clip_output

    def stop_clip(self) -> None:
        clip_output: Optional[FfmpegOutput] = None
        with self._lock:
            if self._clip_output is None:
                return
            clip_output = self._clip_output
            self._clip_output = None
            self._encoder.output = list(self._base_outputs)
        if clip_output is not None:
            self._close_output(clip_output)

    def close(self) -> None:
        with self._lock:
            clip_output = self._clip_output
            self._clip_output = None
            try:
                self._picamera.stop_encoder(self._encoder)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("Failed to stop shared encoder cleanly: %s", exc)
            if clip_output is not None:
                self._close_output(clip_output)

    def _close_output(self, output: FfmpegOutput) -> None:  # type: ignore[name-defined]
        try:
            if hasattr(output, "stop"):
                output.stop()
        except Exception:
            pass
        try:
            if hasattr(output, "close"):
                output.close()
        except Exception:
            pass


class SharedEncoderBackend(RecordingBackend):
    """Recording backend that toggles clip output on a shared encoder."""

    def __init__(self, controller: SharedPicameraEncoder) -> None:
        self._controller = controller

    def start(self, frame: np.ndarray, temp_path: Path, fps: float) -> None:
        del frame, fps
        self._controller.start_clip(temp_path)

    def handle_frame(self, frame: np.ndarray) -> None:
        return

    def stop(self) -> None:
        self._controller.stop_clip()


@dataclass
class ClipSession:
    backend: RecordingBackend
    final_path: Path
    temp_path: Path
    fps: float
    start_time: float
    last_detection_time: float

    def append_frame(self, frame: Any) -> None:
        self.backend.handle_frame(frame)

    def duration(self, now: float) -> float:
        return now - self.start_time

    def since_last_detection(self, now: float) -> float:
        return now - self.last_detection_time

    def mark_detection(self, timestamp: float) -> None:
        self.last_detection_time = timestamp


def timestamp_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def create_clip_session(
    frame: np.ndarray,
    now: float,
    cfg: RecorderConfig,
    backend: RecordingBackend,
    camera_fps: Optional[float],
) -> ClipSession:
    clip_name = f"cat_{timestamp_name()}.mp4"
    final_path = cfg.output_dir / clip_name
    temp_path = cfg.output_dir / f"{final_path.stem}_tmp{final_path.suffix}"

    fps = cfg.fps_override or camera_fps or 30.0
    backend.start(frame, temp_path, fps)

    return ClipSession(
        backend=backend,
        final_path=final_path,
        temp_path=temp_path,
        fps=fps,
        start_time=now,
        last_detection_time=now,
    )


def _promote_clip(temp_path: Path, final_path: Path) -> Path:
    if not temp_path.exists():
        logger.warning("Temp clip missing for %s", final_path)
        return final_path
    try:
        temp_path.rename(final_path)
    except OSError as exc:
        logger.warning("Failed to rename temp clip %s: %s", temp_path, exc)
    return final_path


def finalize_clip(session: ClipSession, reason: str) -> None:
    session.backend.stop()
    saved_path = _promote_clip(session.temp_path, session.final_path)
    logger.info("Recording saved (%s): %s", reason, saved_path)


def discard_clip(session: ClipSession, message: str) -> None:
    session.backend.stop()
    if session.temp_path.exists():
        try:
            session.temp_path.unlink()
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", session.temp_path, exc)
    logger.info("%s", message)


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"camera_index": 0} | (config.get("vision") or {})
    recorder_cfg = RecorderConfig.from_project(config, paths["base"])
    streaming_cfg = config.get("streaming") or {}
    stream_publish_url = streaming_cfg.get("publish_url")
    stream_publish_format = streaming_cfg.get("publish_format", "rtsp")
    stream_publish_options = streaming_cfg.get("publish_options") or {}
    bitrate_value = streaming_cfg.get("bitrate")
    if bitrate_value is None:
        stream_bitrate = recorder_cfg.picamera_bitrate
    else:
        stream_bitrate = int(bitrate_value)

    motion_defaults = {
        "history": 300,
        "var_threshold": 16,
        "dilation_iterations": 2,
        "trigger_ratio": 0.003,
        "min_area": 1500,
        "warmup_frames": 30,
        "blur_kernel": 5,
    }
    motion_cfg = motion_defaults | (config.get("motion") or {})

    resolution_cfg = vision_cfg.get("picamera_resolution")
    if isinstance(resolution_cfg, (list, tuple)) and len(resolution_cfg) >= 2:
        picamera_resolution = (int(resolution_cfg[0]), int(resolution_cfg[1]))
    else:
        picamera_resolution = None
    picamera_fps = float(vision_cfg.get("picamera_fps", 0.0))

    if not (PICAMERA2_AVAILABLE and Picamera2 is not None):
        raise RuntimeError("Picamera2 is required to run the recorder. Install picamera2 on your Pi.")

    lores_resolution = (640, 360)
    camera: CameraInterface = PicameraFrameSource(
        resolution=picamera_resolution,
        preview_resolution=lores_resolution,
        target_fps=picamera_fps,
    )
    shared_encoder: Optional[SharedPicameraEncoder] = None
    backend_factory: Callable[[], RecordingBackend]
    if stream_publish_url:
        try:
            options = {str(k): str(v) for k, v in (stream_publish_options or {}).items()}
            shared_encoder = SharedPicameraEncoder(
                camera.picamera,  # type: ignore[arg-type]
                bitrate=stream_bitrate,
                publish_url=str(stream_publish_url),
                publish_format=str(stream_publish_format),
                publish_options=options,
            )
            backend_factory = lambda: SharedEncoderBackend(shared_encoder)  # type: ignore[misc]
        except Exception as exc:
            logger.error("Unable to start shared encoder for streaming: %s", exc)
            raise
    else:
        backend_factory = lambda: PicameraRecordingBackend(  # type: ignore[attr-defined]
            camera.picamera,
            recorder_cfg.picamera_bitrate,
        )

    try:
        run_motion_recorder(camera, recorder_cfg, motion_cfg, backend_factory)
    finally:
        if shared_encoder:
            shared_encoder.close()
        camera.release()


def run_motion_recorder(
    camera: CameraInterface,
    recorder_cfg: RecorderConfig,
    motion_cfg: Dict[str, Any],
    backend_factory: Callable[[], RecordingBackend],
) -> None:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=int(motion_cfg.get("history", 300)),
        varThreshold=float(motion_cfg.get("var_threshold", 16.0)),
        detectShadows=False,
    )
    dilation_iterations = max(int(motion_cfg.get("dilation_iterations", 2)), 0)
    trigger_ratio = max(float(motion_cfg.get("trigger_ratio", 0.003)), 0.0)
    min_area = max(float(motion_cfg.get("min_area", 1500)), 0.0)
    warmup_frames = max(int(motion_cfg.get("warmup_frames", 30)), 0)
    blur_kernel = int(motion_cfg.get("blur_kernel", 5))
    if blur_kernel < 3:
        blur_kernel = 0
    elif blur_kernel % 2 == 0:
        blur_kernel += 1

    last_clip_time = 0.0
    session: Optional[ClipSession] = None
    frame_counter = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.error("Frame grab failed, exiting.")
                break

            frame_counter += 1
            processed_frame = frame
            if blur_kernel >= 3:
                processed_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
            fg_mask = bg_subtractor.apply(processed_frame)
            _, motion_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            if dilation_iterations > 0:
                motion_mask = cv2.dilate(motion_mask, None, iterations=dilation_iterations)

            had_motion = False
            if frame_counter > warmup_frames:
                frame_area = motion_mask.shape[0] * motion_mask.shape[1]
                motion_ratio = cv2.countNonZero(motion_mask) / float(frame_area or 1)
                if motion_ratio >= trigger_ratio:
                    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) < min_area:
                            continue
                        had_motion = True
                        break

            now = time.time()
            if had_motion:
                if session is None:
                    if (now - last_clip_time) < recorder_cfg.cooldown:
                        continue
                    backend = backend_factory()
                    session = create_clip_session(
                        frame,
                        now,
                        recorder_cfg,
                        backend,
                        camera.fps,
                    )
                    logger.info(
                        "Recording started: %s (target %.0f-%.0fs/%.1f fps)",
                        session.temp_path,
                        recorder_cfg.min_duration,
                        recorder_cfg.max_duration,
                        session.fps,
                    )
                session.append_frame(frame)
                session.mark_detection(now)
                if session.duration(now) >= recorder_cfg.max_duration:
                    finalize_clip(session, "max duration reached")
                    session = None
                    last_clip_time = now
            elif session is not None:
                session.append_frame(frame)
                if session.since_last_detection(now) >= recorder_cfg.absence_grace:
                    if session.duration(now) >= recorder_cfg.min_duration:
                        finalize_clip(session, "motion ended")
                    else:
                        discard_clip(session, "Discarded clip (motion ended before minimum duration).")
                    session = None
                    last_clip_time = now
    except KeyboardInterrupt:
        logger.info("Stopping recorder (Ctrl+C).")
    finally:
        if session is not None:
            if session.duration(time.time()) >= recorder_cfg.min_duration:
                logger.info("Finalizing clip before exit...")
                finalize_clip(session, "manual stop")
            else:
                discard_clip(session, "Discarding unfinished clip (insufficient duration).")
            session = None


class PicameraFrameSource(CameraInterface):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        preview_resolution: Optional[Tuple[int, int]],
        target_fps: float,
    ) -> None:
        if Picamera2 is None:
            raise CameraError("Picamera2 is not available on this system.")
        self._picam = Picamera2()
        main_config: Dict[str, Any] = {"format": "YUV420"}
        if resolution:
            main_config["size"] = resolution
        lores_size = preview_resolution or resolution or (640, 360)
        self._lores_size = (int(lores_size[0]), int(lores_size[1]))
        lores_config: Dict[str, Any] = {
            "format": "YUV420",
            "size": self._lores_size,
        }
        controls: Dict[str, Any] = {}
        self._fps = float(target_fps) if target_fps and target_fps > 0 else 0.0
        if self._fps > 0:
            frame_duration = int(1_000_000 / self._fps)
            controls["FrameDurationLimits"] = (frame_duration, frame_duration)
        config = self._picam.create_video_configuration(
            main=main_config,
            lores=lores_config,
            controls=controls,
            buffer_count=6,
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.05)
        if self._fps <= 0:
            self._fps = self._measure_running_fps()
        if self._fps <= 0:
            self._fps = 30.0
        self._open = True

    def read(self) -> Tuple[bool, np.ndarray]:
        try:
            frame = self._picam.capture_array("lores")
        except Exception as exc:
            logger.error("Failed to capture frame from Picamera2: %s", exc)
            return False, np.zeros((1, 1, 3), dtype=np.uint8)
        bgr = self._convert_to_bgr(frame)
        return True, bgr

    def release(self) -> None:
        if not self._open:
            return
        try:
            self._picam.close()
        finally:
            self._open = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def picamera(self) -> Picamera2:  # type: ignore[valid-type]
        return self._picam

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

    def _convert_to_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            if channels == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if channels == 2:
                return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        if frame.ndim == 2:
            width, height = self._lores_size
            expected_rows = int(height * 1.5)
            if frame.shape[0] == expected_rows and frame.shape[1] == width:
                return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        frame = np.atleast_3d(frame)
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        return frame[:, :, :3]


if __name__ == "__main__":
    configure_logging()
    main()
