from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from cat_face.detection import create_detector
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
    show_window: bool

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
            "show_window": False,
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
            show_window=bool(raw_cfg.get("show_window", False)),
        )


@dataclass
class ClipSession:
    writer: cv2.VideoWriter
    final_path: Path
    temp_path: Path
    fps: float
    start_time: float
    last_detection_time: float
    frames: List[Any] = field(default_factory=list)

    def append_frame(self, frame: Any) -> None:
        self.frames.append(frame.copy())

    def duration(self, now: float) -> float:
        return now - self.start_time

    def since_last_detection(self, now: float) -> float:
        return now - self.last_detection_time

    def mark_detection(self, timestamp: float) -> None:
        self.last_detection_time = timestamp

    def clear_frames(self) -> None:
        self.frames.clear()


def timestamp_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def create_clip_session(
    frame: Any,
    now: float,
    cfg: RecorderConfig,
    fourcc: int,
    cap: cv2.VideoCapture,
) -> ClipSession:
    clip_name = f"cat_{timestamp_name()}.mp4"
    final_path = cfg.output_dir / clip_name
    temp_path = cfg.output_dir / f"{final_path.stem}_tmp{final_path.suffix}"

    frame_height, frame_width = frame.shape[:2]
    fps = cfg.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError("Unable to create video writer for clip recording.")

    return ClipSession(
        writer=writer,
        final_path=final_path,
        temp_path=temp_path,
        fps=fps,
        start_time=now,
        last_detection_time=now,
    )


def _write_frames(session: ClipSession) -> None:
    for frame in session.frames:
        session.writer.write(frame)
    session.clear_frames()


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
    _write_frames(session)
    session.writer.release()
    saved_path = _promote_clip(session.temp_path, session.final_path)
    print(f"Recording saved ({reason}): {saved_path}")


def discard_clip(session: ClipSession, message: str) -> None:
    session.writer.release()
    session.clear_frames()
    if session.temp_path.exists():
        session.temp_path.unlink(missing_ok=True)
    print(message)


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"camera_index": 0} | (config.get("vision") or {})
    recorder_cfg = RecorderConfig.from_project(config, paths["base"])

    camera_index = int(vision_cfg["camera_index"])
    detector = create_detector(config.get("detection"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    fourcc = cv2.VideoWriter_fourcc(*recorder_cfg.codec)

    print("Press Q to exit.")
    last_clip_time = 0.0
    session: Optional[ClipSession] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            now = time.time()
            detection_boxes = detector.detect(frame)
            had_detection = len(detection_boxes) > 0

            if recorder_cfg.show_window:
                preview = frame.copy()
                for (x, y, w, h) in detection_boxes:
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Cat Clip Recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if had_detection:
                if session is None:
                    if (now - last_clip_time) < recorder_cfg.cooldown:
                        continue
                    session = create_clip_session(frame, now, recorder_cfg, fourcc, cap)
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
                        finalize_clip(session, "cat left frame")
                    else:
                        discard_clip(session, "Discarded clip (cat left before minimum duration).")
                    session = None
                    last_clip_time = now

    finally:
        if session is not None:
            if session.duration(time.time()) >= recorder_cfg.min_duration:
                print("Finalizing clip before exit...")
                finalize_clip(session, "manual stop")
            else:
                discard_clip(session, "Discarding unfinished clip (insufficient duration).")
            session = None
        if cap.isOpened():
            cap.release()
        if recorder_cfg.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
