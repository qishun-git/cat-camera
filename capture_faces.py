from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

from cat_face.detection import create_detector
from cat_face.utils import ensure_dir, load_project_config, preprocess_face, resolve_paths, rotate_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture cat faces for labeling or training.")
    parser.add_argument(
        "--cat-name",
        help="Name of the cat when running in labeled mode. Required for labeled capture sessions.",
    )
    return parser.parse_args()


def load_capture_settings(cli_cat_name: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Path], Optional[str], Dict[str, Any]]:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_defaults = {
        "camera_index": 0,
        "face_size": 100,
    }
    vision_cfg = vision_defaults | config.get("vision", {})
    defaults = {
        "mode": "labeled",
        "display_window": True,
        "auto_session_subfolders": True,
        "max_images_per_cat": 500,
        "max_unlabeled_images": 1000,
    }
    capture_cfg = defaults | config.get("capture", {})
    capture_cfg["camera_index"] = vision_cfg["camera_index"]
    capture_cfg["size"] = vision_cfg["face_size"]
    configured_name = capture_cfg.get("cat_name")
    active_cat_name = cli_cat_name or configured_name
    return capture_cfg, paths, active_cat_name, config


def resolve_output(cfg: Dict[str, Any], paths: Dict[str, Path], cat_name: Optional[str]) -> Path:
    mode = cfg["mode"]
    if mode == "unlabeled":
        base_output = ensure_dir(paths["unlabeled"])
        if cfg.get("auto_session_subfolders", True):
            session_name = datetime.now().strftime("session-%Y%m%d-%H%M%S")
            return ensure_dir(base_output / session_name)
        return base_output
    target_root = ensure_dir(paths["training"])
    target_name = cat_name or "unknown"
    return ensure_dir(target_root / target_name)


def main() -> None:
    args = parse_args()
    cfg, paths, cat_name, config = load_capture_settings(args.cat_name)
    output_dir = resolve_output(cfg, paths, cat_name)
    detector = create_detector(config.get("detection"))
    cap = cv2.VideoCapture(int(cfg["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {cfg['camera_index']}")

    mode = cfg["mode"]
    if mode == "labeled":
        if not cat_name:
            raise ValueError("Capture mode 'labeled' requires --cat-name to be specified.")
        label = cat_name
    else:
        label = "unlabeled"
    print("Press SPACE to save detected faces, or Q to quit.")
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            faces = detector.detect(frame)

            if cfg["display_window"]:
                display = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Cat Capture", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = None

            should_save = len(faces) > 0
            if key == ord(" "):
                should_save = True
            if key == ord("q"):
                break

            if should_save and len(faces) > 0:
                timestamp = int(time.time() * 1000)
                for idx, (x, y, w, h) in enumerate(faces):
                    color_crop = frame[y : y + h, x : x + w]
                    processed = preprocess_face(color_crop, size=(int(cfg["size"]), int(cfg["size"])))
                    filename = output_dir / f"{label}_{timestamp}_{idx}.png"
                    cv2.imwrite(str(filename), processed)
                    saved += 1
                limit = int(cfg["max_images_per_cat"] if mode == "labeled" else cfg["max_unlabeled_images"])
                removed = rotate_files(output_dir, limit)
                if removed:
                    print(f"Rotation: removed {removed} oldest image(s) from {output_dir}")
                print(f"Saved {len(faces)} face(s). Total images: {saved}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
