from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from cat_face.camera import CameraError, Picamera2Camera, create_camera
from cat_face.streamer import MJPEGStreamer
from cat_face.utils import ensure_dir, load_project_config, resolve_paths


@dataclass
class RecorderConfig:
    output_dir: Path
    min_duration: float
    max_duration: float
    cooldown: float
    absence_grace: float
    fps_override: Optional[float]
    codec: str
    native_bitrate: int

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
            "native_bitrate": 4000000,
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
            native_bitrate=int(raw_cfg.get("native_bitrate", 4_000_000)),
        )


class NativeRecorder:
    def __init__(self, camera: Picamera2Camera, bitrate: int = 4_000_000) -> None:
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput

        self._camera = camera
        self._bitrate = bitrate
        self._H264Encoder = H264Encoder
        self._FfmpegOutput = FfmpegOutput
        self._active = False

    @property
    def fps(self) -> float:
        return self._camera.fps

    def start(self, output_path: Path) -> None:
        if self._active:
            raise RuntimeError("Native recorder already running.")
        encoder = self._H264Encoder(bitrate=self._bitrate)
        output = self._FfmpegOutput(str(output_path))
        self._encoder = encoder
        self._output = output
        self._camera.raw.start_recording(encoder, output)
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._camera.raw.stop_recording()
        self._encoder = None
        self._output = None
        self._active = False


@dataclass
class ClipSession:
    writer: Optional[cv2.VideoWriter]
    final_path: Path
    temp_path: Path
    fps: float
    start_time: float
    last_detection_time: float
    native_recorder: Optional[NativeRecorder] = None
    using_native: bool = False

    def append_frame(self, frame: Any) -> None:
        if self.writer:
            self.writer.write(frame)

    def duration(self, now: float) -> float:
        return now - self.start_time

    def since_last_detection(self, now: float) -> float:
        return now - self.last_detection_time

    def mark_detection(self, timestamp: float) -> None:
        self.last_detection_time = timestamp


def timestamp_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def create_clip_session(
    frame: Any,
    now: float,
    cfg: RecorderConfig,
    fourcc: int,
    camera_fps: Optional[float],
    native_recorder: Optional[NativeRecorder] = None,
) -> ClipSession:
    clip_name = f"cat_{timestamp_name()}.mp4"
    final_path = cfg.output_dir / clip_name
    temp_path = cfg.output_dir / f"{final_path.stem}_tmp{final_path.suffix}"

    frame_height, frame_width = frame.shape[:2]
    fps = cfg.fps_override or camera_fps or 30.0
    if native_recorder:
        native_recorder.start(temp_path)
        if native_recorder.fps > 0:
            fps = native_recorder.fps
        writer = None
        using_native = True
    else:
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise RuntimeError("Unable to create video writer for clip recording.")
        using_native = False

    return ClipSession(
        writer=writer,
        final_path=final_path,
        temp_path=temp_path,
        fps=fps,
        start_time=now,
        last_detection_time=now,
        native_recorder=native_recorder if using_native else None,
        using_native=using_native,
    )


def _promote_clip(temp_path: Path, final_path: Path) -> Path:
    if not temp_path.exists():
        print(f"Warning: temp clip missing for {final_path}")
        return final_path
    try:
        temp_path.rename(final_path)
    except OSError as exc:
        print(f"Warning: failed to rename temp clip {temp_path}: {exc}")
    return final_path


def finalize_clip(session: ClipSession, reason: str) -> None:
    if session.using_native and session.native_recorder:
        session.native_recorder.stop()
    if session.writer:
        session.writer.release()
    saved_path = _promote_clip(session.temp_path, session.final_path)
    print(f"Recording saved ({reason}): {saved_path}")


def discard_clip(session: ClipSession, message: str) -> None:
    if session.using_native and session.native_recorder:
        session.native_recorder.stop()
    if session.writer:
        session.writer.release()
    if session.temp_path.exists():
        session.temp_path.unlink(missing_ok=True)
    print(message)


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"camera_index": 0} | (config.get("vision") or {})
    recorder_cfg = RecorderConfig.from_project(config, paths["base"])
    streaming_cfg = config.get("streaming") or {}
    stream_resolution = streaming_cfg.get("resolution")
    if isinstance(stream_resolution, (list, tuple)) and len(stream_resolution) >= 2:
        stream_resolution = (int(stream_resolution[0]), int(stream_resolution[1]))
    else:
        stream_resolution = None
    stream_quality = int(streaming_cfg.get("quality", 80))
    status_path = streaming_cfg.get("status_path")
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

    camera_index = int(vision_cfg["camera_index"])
    prefer_picamera = bool(vision_cfg.get("prefer_picamera2", False))
    resolution_cfg = vision_cfg.get("picamera_resolution")
    if isinstance(resolution_cfg, (list, tuple)) and len(resolution_cfg) >= 2:
        picamera_resolution = (int(resolution_cfg[0]), int(resolution_cfg[1]))
    else:
        picamera_resolution = None
    picamera_fps = vision_cfg.get("picamera_fps")
    picamera_fps = float(picamera_fps) if picamera_fps else None
    native_recorder: Optional[NativeRecorder] = None
    try:
        camera = create_camera(
            camera_index=camera_index,
            prefer_picamera=prefer_picamera,
            picamera_resolution=picamera_resolution,
            picamera_fps=picamera_fps,
        )
    except CameraError as exc:
        raise RuntimeError(str(exc)) from exc
    if prefer_picamera and isinstance(camera, Picamera2Camera):
        try:
            native_recorder = NativeRecorder(camera, bitrate=recorder_cfg.native_bitrate)
        except Exception as exc:
            print(f"Warning: Picamera2 native recorder unavailable ({exc}). Falling back to OpenCV writer.")
            native_recorder = None

    fourcc = cv2.VideoWriter_fourcc(*recorder_cfg.codec)

    streamer: Optional[MJPEGStreamer] = None
    try:
        streamer = MJPEGStreamer(
            host=str(streaming_cfg.get("host", "0.0.0.0")),
            port=int(streaming_cfg.get("port", 8765)),
            resolution=stream_resolution,
            quality=stream_quality,
            frame_interval=float(streaming_cfg.get("frame_interval", 0.03)),
            status_path=str(status_path) if status_path else None,
        )
    except OSError as exc:
        print(f"Warning: unable to start MJPEG streamer: {exc}")
        streamer = None

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
                print("Frame grab failed, exiting.")
                break

            frame_counter += 1
            processed_frame = frame
            if blur_kernel >= 3:
                processed_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
            fg_mask = bg_subtractor.apply(processed_frame)
            _, motion_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            if dilation_iterations > 0:
                motion_mask = cv2.dilate(motion_mask, None, iterations=dilation_iterations)

            motion_boxes: List[Tuple[int, int, int, int]] = []
            had_motion = False
            if frame_counter > warmup_frames:
                frame_area = motion_mask.shape[0] * motion_mask.shape[1]
                motion_ratio = cv2.countNonZero(motion_mask) / float(frame_area or 1)
                if motion_ratio >= trigger_ratio:
                    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) < min_area:
                            continue
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_boxes.append((x, y, w, h))
                had_motion = bool(motion_boxes)

            if streamer:
                streamer.push_frame(frame)

            now = time.time()
            if had_motion:
                if session is None:
                    if (now - last_clip_time) < recorder_cfg.cooldown:
                        continue
                    session = create_clip_session(
                        frame,
                        now,
                        recorder_cfg,
                        fourcc,
                        camera.fps,
                        native_recorder=native_recorder,
                    )
                    print(
                        f"Recording started: {session.temp_path} "
                        f"(target {recorder_cfg.min_duration:.0f}-{recorder_cfg.max_duration:.0f}s/{session.fps:.1f} fps)"
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

    finally:
        if streamer:
            streamer.stop()
        if session is not None:
            if session.duration(time.time()) >= recorder_cfg.min_duration:
                print("Finalizing clip before exit...")
                finalize_clip(session, "manual stop")
            else:
                discard_clip(session, "Discarding unfinished clip (insufficient duration).")
            session = None
        camera.release()


if __name__ == "__main__":
    main()
