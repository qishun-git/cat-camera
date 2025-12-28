from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, compute_centroids
from cat_face.utils import (
    ensure_dir,
    iter_image_files,
    load_project_config,
    preprocess_face,
    resolve_paths,
    save_label_map,
)


def load_dataset(data_dir: Path, size: int) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[int, str] = {}

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {data_dir}")

    for idx, class_dir in enumerate(class_dirs):
        label_map[idx] = class_dir.name
        for img_path in iter_image_files(class_dir):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: unable to read {img_path}, skipping.")
                continue
            processed = preprocess_face(img, size=(size, size))
            images.append(processed)
            labels.append(idx)

    if len(images) < 2:
        raise RuntimeError("Need at least two images to train.")
    return images, labels, label_map


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"face_size": 100} | config.get("vision", {})
    training_defaults = {
        "embedding_model_filename": "embeddings.npz",
        "embedding_input_size": 224,
    }
    train_cfg = training_defaults | config.get("training", {})

    data_dir = Path(train_cfg.get("data_dir", paths["training"]))
    images, labels, label_map = load_dataset(data_dir, int(vision_cfg["face_size"]))

    extractor = EmbeddingExtractor(input_size=int(train_cfg["embedding_input_size"]))
    embeddings: List[np.ndarray] = []
    for img in images:
        embeddings.append(extractor.extract(img))

    model = compute_centroids(embeddings, labels)

    models_dir = ensure_dir(paths["models"])
    model_path = models_dir / train_cfg["embedding_model_filename"]
    labels_path = models_dir / train_cfg.get("labels_filename", "labels.json")
    model.save(model_path)
    save_label_map(label_map, labels_path)

    print(f"Embedding model saved to {model_path}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
