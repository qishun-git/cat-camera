from __future__ import annotations

import math
from typing import Dict

import cv2

from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer
from cat_face.detection import create_detector
from cat_face.utils import load_label_map, load_project_config, preprocess_face, resolve_paths


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_defaults: Dict[str, object] = {
        "camera_index": 0,
        "face_size": 100,
    }
    vision_cfg = vision_defaults | config.get("vision", {})
    training_cfg = config.get("training", {})
    recog_defaults: Dict[str, object] = {
        "threshold": training_cfg.get("threshold", 80.0),
        "embedding_threshold": 0.75,
        "embedding_model_filename": training_cfg.get("embedding_model_filename", "embeddings.npz"),
    }
    recog_cfg = recog_defaults | config.get("recognition", {})

    recog_cfg.setdefault("camera_index", vision_cfg["camera_index"])
    recog_cfg.setdefault("size", vision_cfg["face_size"])

    method = str(recog_cfg.get("method", "lbph")).lower()
    model_filename = None
    if method == "lbph":
        model_filename = recog_cfg.get("model_filename") or training_cfg.get("model_filename", "lbph_model.xml")
    embedding_model_filename = recog_cfg.get("embedding_model_filename") or training_cfg.get("embedding_model_filename", "embeddings.npz")
    labels_filename = recog_cfg.get("labels_filename") or training_cfg.get("labels_filename", "labels.json")

    models_dir = paths["models"]
    labels_path = models_dir / labels_filename
    if not labels_path.exists():
        raise FileNotFoundError(f"Label map not found: {labels_path}")

    labels = load_label_map(labels_path)
    embed_recognizer: EmbeddingRecognizer | None = None
    recognizer = None
    if method == "embedding":
        embedding_path = models_dir / embedding_model_filename
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {embedding_path}")
        extractor = EmbeddingExtractor(input_size=int(training_cfg.get("embedding_input_size", 224)))
        embedding_model = EmbeddingModel.load(embedding_path)
        embed_recognizer = EmbeddingRecognizer(
            model=embedding_model,
            extractor=extractor,
            threshold=float(recog_cfg.get("embedding_threshold", 0.75)),
        )
    else:
        model_filename = model_filename or training_cfg.get("model_filename", "lbph_model.xml")
        model_path = models_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model_path))
        recognizer.setThreshold(float(recog_cfg["threshold"]))

    detector = create_detector(config.get("detection"))
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
            faces = detector.detect(frame)

            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                processed = preprocess_face(roi, size=(int(recog_cfg["size"]), int(recog_cfg["size"])))
                if method == "embedding":
                    assert embed_recognizer is not None
                    label_id, score = embed_recognizer.predict(processed)
                    if label_id == -1:
                        name = "unknown"
                        color = (0, 0, 255)
                    else:
                        name = labels.get(label_id, "unknown")
                        color = (0, 255, 0)
                    conf_text = f"{score:.2f}"
                else:
                    assert recognizer is not None
                    label_id, confidence = recognizer.predict(processed)
                    is_conf_finite = math.isfinite(confidence)
                    threshold = float(recog_cfg["threshold"])
                    if label_id == -1 or not is_conf_finite:
                        name = "unknown"
                        color = (0, 0, 255)
                        conf_text = "N/A"
                    else:
                        if confidence <= threshold:
                            name = labels.get(label_id, "unknown")
                            color = (0, 255, 0)
                        else:
                            name = "unknown"
                            color = (0, 0, 255)
                        conf_text = f"{confidence:.1f}"
                text = f"{name} ({conf_text})"
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
