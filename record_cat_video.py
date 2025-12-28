from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2

from cat_face.detection import create_detector
from cat_face.utils import ensure_dir, load_project_config, resolve_paths


def timestamp_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"camera_index": 0} | config.get("vision", {})
    recorder_defaults = {
        "output_dir": str(paths["base"] / "clips"),
        "min_duration": 15.0,
        "max_duration": 30.0,
        "cooldown": 5.0,
        "absence_grace": 1.0,
        "fps": 0.0,
        "codec": "mp4v",
        "show_window": False,
    }
    recorder_cfg: Dict[str, object] = recorder_defaults | config.get("recorder", {})

    camera_index = int(vision_cfg["camera_index"])
    detector = create_detector(config.get("detection"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    output_root = ensure_dir(Path(str(recorder_cfg["output_dir"])))

    enforced_min = max(float(recorder_cfg["min_duration"]), 0.0)
    enforced_max = max(float(recorder_cfg["max_duration"]), enforced_min)
    cooldown = max(float(recorder_cfg["cooldown"]), 0.0)
    absence_grace = max(float(recorder_cfg.get("absence_grace", 1.0)), 0.0)

    window_enabled = bool(recorder_cfg.get("show_window", False))

    codec_value = str(recorder_cfg.get("codec", "mp4v"))
    fourcc = cv2.VideoWriter_fourcc(*codec_value)

    print("Press Q to exit.")
    last_clip_time = 0.0
    writer: Optional[cv2.VideoWriter] = None
    clip_frames: Deque[Tuple[float, any]] = deque()
    clip_start_time = 0.0
    clip_path: Optional[Path] = None
    clip_temp_path: Optional[Path] = None
    last_detection_time = 0.0
    fps_cfg = float(recorder_cfg.get("fps", 0.0))
    fps_override = fps_cfg if fps_cfg and fps_cfg > 0 else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            now = time.time()
            detection_boxes = detector.detect(frame)
            had_detection = len(detection_boxes) > 0

            if window_enabled:
                preview = frame.copy()
                for (x, y, w, h) in detection_boxes:
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Cat Clip Recorder", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if had_detection:
                if writer is None:
                    if (now - last_clip_time) >= cooldown:
                        clip_start_time = now
                        last_detection_time = now
                        clip_name = f"cat_{timestamp_name()}.mp4"
                        clip_path = output_root / clip_name
                        clip_temp_path = output_root / f"{clip_path.stem}_tmp{clip_path.suffix}"
                        frame_height, frame_width = frame.shape[:2]
                        fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
                        writer = cv2.VideoWriter(str(clip_temp_path), fourcc, fps, (frame_width, frame_height))
                        clip_frames.clear()
                        print(f"Recording started: {clip_temp_path} (target {enforced_min:.0f}-{enforced_max:.0f}s/{fps} fps)")
                    else:
                        continue
                clip_frames.append((now, frame.copy()))
                last_detection_time = now
                if now - clip_start_time >= enforced_max:
                    for _, buffered_frame in clip_frames:
                        writer.write(buffered_frame)
                    writer.release()
                    writer = None
                    if clip_temp_path and clip_temp_path.exists():
                        try:
                            clip_temp_path.rename(clip_path)
                        except OSError as exc:
                            print(f"Warning: failed to rename temp clip {clip_temp_path}: {exc}")
                    last_clip_time = now
                    print(f"Recording saved (max duration reached): {clip_path}")
                    clip_path = None
                    clip_temp_path = None
                    clip_frames.clear()
            elif writer is not None:
                clip_frames.append((now, frame.copy()))
                no_detection_elapsed = now - last_detection_time
                clip_duration = now - clip_start_time
                if no_detection_elapsed >= absence_grace:
                    if clip_duration >= enforced_min:
                        for _, buffered_frame in clip_frames:
                            writer.write(buffered_frame)
                        writer.release()
                        writer = None
                        if clip_temp_path and clip_temp_path.exists():
                            try:
                                clip_temp_path.rename(clip_path)
                            except OSError as exc:
                                print(f"Warning: failed to rename temp clip {clip_temp_path}: {exc}")
                        last_clip_time = now
                        print(f"Recording saved (cat left frame): {clip_path}")
                        clip_path = None
                        clip_temp_path = None
                        clip_frames.clear()
                    else:
                        writer.release()
                        writer = None
                        if clip_temp_path and clip_temp_path.exists():
                            clip_temp_path.unlink(missing_ok=True)
                        print("Discarded clip (cat left before minimum duration).")
                        clip_path = None
                        clip_temp_path = None
                        clip_frames.clear()
                    last_clip_time = now
            else:
                clip_frames.clear()

    finally:
        if writer is not None:
            if writer.isOpened() and clip_frames and (time.time() - clip_start_time) >= enforced_min:
                print("Finalizing clip before exit...")
                for _, buffered_frame in clip_frames:
                    writer.write(buffered_frame)
                    if clip_temp_path and clip_temp_path.exists():
                        try:
                            clip_temp_path.rename(clip_path)
                        except OSError as exc:
                            print(f"Warning: failed to rename temp clip {clip_temp_path}: {exc}")
                print(f"Recording saved: {clip_path}")
            else:
                print("Discarding unfinished clip (insufficient duration).")
                if clip_temp_path and clip_temp_path.exists():
                    clip_temp_path.unlink(missing_ok=True)
            writer.release()
        clip_path = None
        clip_temp_path = None
        if cap.isOpened():
            cap.release()
        if window_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
