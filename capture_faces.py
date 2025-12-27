from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2

from cat_face.utils import CONFIG_DIR, DATA_DIR, default_cascade_path, ensure_dir, load_yaml, preprocess_face, rotate_files

CONFIG_PATH = CONFIG_DIR / "capture.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    cfg = load_yaml(path)
    cfg.setdefault("capture_mode", "labeled")
    cfg.setdefault("cat_name", "unknown")
    cfg.setdefault("output", str(DATA_DIR))
    cfg.setdefault("camera_index", 0)
    cfg.setdefault("cascade", None)
    cfg.setdefault("size", 100)
    cfg.setdefault("scale_factor", 1.1)
    cfg.setdefault("min_neighbors", 3)
    cfg.setdefault("min_size", 60)
    cfg.setdefault("display_window", True)
    cfg.setdefault("auto_session_subfolders", True)
    cfg.setdefault("max_images_per_cat", 500)
    cfg.setdefault("max_unlabeled_images", 1000)
    return cfg


def resolve_output(cfg: Dict[str, Any]) -> Path:
    base_output = ensure_dir(Path(cfg["output"]))
    mode = cfg["capture_mode"]
    if mode == "unlabeled":
        if cfg.get("auto_session_subfolders", True):
            session_name = datetime.now().strftime("session-%Y%m%d-%H%M%S")
            return ensure_dir(base_output / "unlabeled" / session_name)
        return ensure_dir(base_output / "unlabeled")
    return ensure_dir(base_output / cfg["cat_name"])


def main() -> None:
    cfg = load_config()
    output_dir = resolve_output(cfg)
    cascade_path = Path(cfg["cascade"]) if cfg.get("cascade") else default_cascade_path()
    if not cascade_path.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))
    cap = cv2.VideoCapture(int(cfg["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {cfg['camera_index']}")

    mode = cfg["capture_mode"]
    label = cfg["cat_name"] if mode == "labeled" else "unlabeled"
    print("Press SPACE to save detected faces, or Q to quit.")
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=float(cfg["scale_factor"]),
                minNeighbors=int(cfg["min_neighbors"]),
                minSize=(int(cfg["min_size"]), int(cfg["min_size"])),
            )

            if cfg["display_window"]:
                display = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Cat Capture", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = None

            should_save = not cfg["display_window"] and len(faces) > 0
            if key == ord(" "):
                should_save = True
            if key == ord("q"):
                break

            if should_save and len(faces) > 0:
                timestamp = int(time.time() * 1000)
                for idx, (x, y, w, h) in enumerate(faces):
                    cropped = gray[y : y + h, x : x + w]
                    processed = preprocess_face(cropped, size=(int(cfg["size"]), int(cfg["size"])))
                    filename = output_dir / f"{label}_{timestamp}_{idx}.png"
                    cv2.imwrite(str(filename), processed)
                    saved += 1
                if mode == "labeled":
                    limit = int(cfg["max_images_per_cat"])
                else:
                    limit = int(cfg["max_unlabeled_images"])
                removed = rotate_files(output_dir, limit)
                if removed:
                    print(f"Rotation: removed {removed} oldest image(s) from {output_dir}")
                print(f"Saved {len(faces)} face(s). Total images: {saved}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
