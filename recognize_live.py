from __future__ import annotations

import logging
from typing import Dict

import cv2

from cat_face.detection import create_detector
from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer
from cat_face.utils import configure_logging, load_label_map, load_project_config, preprocess_face, resolve_paths

logger = logging.getLogger(__name__)


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
        "embedding_threshold": 0.75,
        "embedding_model_filename": training_cfg.get("embedding_model_filename", "embeddings.npz"),
        "labels_filename": training_cfg.get("labels_filename", "labels.json"),
    }
    recog_cfg = recog_defaults | config.get("recognition", {})

    models_dir = paths["models"]
    labels_path = models_dir / recog_cfg["labels_filename"]
    if not labels_path.exists():
        raise FileNotFoundError(f"Label map not found: {labels_path}")

    embedding_path = models_dir / recog_cfg["embedding_model_filename"]
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding model not found: {embedding_path}")

    labels = load_label_map(labels_path)
    extractor = EmbeddingExtractor(input_size=int(training_cfg.get("embedding_input_size", 224)))
    embed_recognizer = EmbeddingRecognizer(
        model=EmbeddingModel.load(embedding_path),
        extractor=extractor,
        threshold=float(recog_cfg.get("embedding_threshold", 0.75)),
    )

    detector = create_detector(config.get("detection"))
    camera_index = int(vision_cfg["camera_index"])
    face_size = int(vision_cfg["face_size"])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    logger.info("Press Q to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame grab failed, exiting.")
                break

            faces = detector.detect(frame)

            for (x, y, w, h) in faces:
                roi = frame[y : y + h, x : x + w]
                processed = preprocess_face(roi, size=(face_size, face_size))
                label_id, score = embed_recognizer.predict(processed)
                if label_id == -1:
                    name = "unknown"
                    color = (0, 0, 255)
                else:
                    name = labels.get(label_id, "unknown")
                    color = (0, 255, 0)
                conf_text = f"{score:.2f}"
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
    configure_logging()
    main()
