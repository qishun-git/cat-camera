from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2

from cat_face.utils import (
    default_cascade_path,
    load_label_map,
    load_project_config,
    preprocess_face,
    resolve_paths,
)


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    shared_capture_defaults: Dict[str, object] = {
        "camera_index": 0,
        "cascade": "",
        "size": 100,
        "scale_factor": 1.1,
        "min_neighbors": 3,
        "min_size": 60,
    }
    capture_cfg = shared_capture_defaults | config.get("capture", {})
    training_cfg = config.get("training", {})
    recog_defaults: Dict[str, object] = {
        "threshold": training_cfg.get("threshold", 80.0),
    }
    recog_cfg = recog_defaults | config.get("recognition", {})

    for key in shared_capture_defaults:
        recog_cfg.setdefault(key, capture_cfg[key])

    model_filename = recog_cfg.get("model_filename") or training_cfg.get("model_filename", "lbph_model.xml")
    labels_filename = recog_cfg.get("labels_filename") or training_cfg.get("labels_filename", "labels.json")

    models_dir = paths["models"]
    model_path = models_dir / model_filename
    labels_path = models_dir / labels_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label map not found: {labels_path}")

    labels = load_label_map(labels_path)
    cascade_path = Path(recog_cfg["cascade"]) if recog_cfg.get("cascade") else default_cascade_path()
    if not cascade_path.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))
    recognizer.setThreshold(float(recog_cfg["threshold"]))

    detector = cv2.CascadeClassifier(str(cascade_path))
    cap = cv2.VideoCapture(int(recog_cfg["camera_index"]))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {recog_cfg['camera_index']}")

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
                scaleFactor=float(recog_cfg["scale_factor"]),
                minNeighbors=int(recog_cfg["min_neighbors"]),
                minSize=(int(recog_cfg["min_size"]), int(recog_cfg["min_size"])),
            )

            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                processed = preprocess_face(roi, size=(int(recog_cfg["size"]), int(recog_cfg["size"])))
                label_id, confidence = recognizer.predict(processed)
                if confidence <= float(recog_cfg["threshold"]):
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
