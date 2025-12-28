from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

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
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: unable to read {img_path}, skipping.")
                continue
            images.append(preprocess_face(img, size=(size, size)))
            labels.append(idx)

    if len(images) < 2:
        raise RuntimeError("Need at least two images to train.")
    return images, labels, label_map


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    defaults = {
        "size": 100,
        "radius": 2,
        "neighbors": 8,
        "grid_x": 8,
        "grid_y": 8,
        "threshold": 80.0,
        "model_filename": "lbph_model.xml",
        "labels_filename": "labels.json",
    }
    train_cfg = defaults | config.get("training", {})

    data_dir = Path(train_cfg.get("data_dir", paths["training"]))
    images, labels, label_map = load_dataset(data_dir, int(train_cfg["size"]))

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=int(train_cfg["radius"]),
        neighbors=int(train_cfg["neighbors"]),
        grid_x=int(train_cfg["grid_x"]),
        grid_y=int(train_cfg["grid_y"]),
    )
    recognizer.setThreshold(float(train_cfg["threshold"]))
    recognizer.train(images, np.array(labels))

    models_dir = ensure_dir(paths["models"])
    model_path = models_dir / train_cfg["model_filename"]
    labels_path = models_dir / train_cfg["labels_filename"]
    recognizer.write(str(model_path))
    save_label_map(label_map, labels_path)

    print(f"Model saved to {model_path}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
