from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2

from cat_face.utils import CONFIG_DIR, MODEL_DIR, default_cascade_path, load_label_map, load_yaml, preprocess_face

CONFIG_PATH = CONFIG_DIR / "recognize.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    cfg = load_yaml(path)
    cfg.setdefault("model_path", str(MODEL_DIR / "lbph_model.xml"))
    cfg.setdefault("labels_path", str(MODEL_DIR / "labels.json"))
    cfg.setdefault("camera_index", 0)
    cfg.setdefault("cascade", None)
    cfg.setdefault("size", 100)
    cfg.setdefault("threshold", 80.0)
    cfg.setdefault("scale_factor", 1.1)
    cfg.setdefault("min_neighbors", 3)
    cfg.setdefault("min_size", 60)
    return cfg


def main() -> None:
    cfg = load_config()
    model_path = Path(cfg["model_path"])
    labels_path = Path(cfg["labels_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label map not found: {labels_path}")

    labels = load_label_map(labels_path)
    cascade_path = Path(cfg["cascade"]) if cfg.get("cascade") else default_cascade_path()
    if not cascade_path.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    recognizer.setThreshold(float(cfg["threshold"]))

    detector = cv2.CascadeClassifier(str(cascade_path))
    cap = cv2.VideoCapture(int(cfg["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {cfg['camera_index']}")

    print("Press Q to exit.")
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

            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                processed = preprocess_face(roi, size=(int(cfg["size"]), int(cfg["size"])))
                label_id, confidence = recognizer.predict(processed)
                if confidence <= float(cfg["threshold"]):
                    name = labels.get(label_id, "unknown")
                    color = (0, 255, 0)
                else:
                    name = "unknown"
                    color = (0, 0, 255)
                text = f"{name} ({confidence:.1f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Cat Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
