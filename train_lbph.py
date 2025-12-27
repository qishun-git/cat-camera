from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from cat_face.utils import (
    CONFIG_DIR,
    DATA_DIR,
    MODEL_DIR,
    ensure_dir,
    iter_image_files,
    load_yaml,
    preprocess_face,
    save_label_map,
)

CONFIG_PATH = CONFIG_DIR / "train.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    cfg = load_yaml(path)
    cfg.setdefault("data_dir", str(DATA_DIR))
    cfg.setdefault("model_path", str(MODEL_DIR / "lbph_model.xml"))
    cfg.setdefault("labels_path", str(MODEL_DIR / "labels.json"))
    cfg.setdefault("size", 100)
    cfg.setdefault("radius", 2)
    cfg.setdefault("neighbors", 8)
    cfg.setdefault("grid_x", 8)
    cfg.setdefault("grid_y", 8)
    cfg.setdefault("threshold", 80.0)
    return cfg


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
    cfg = load_config()
    data_dir = Path(cfg["data_dir"])
    images, labels, label_map = load_dataset(data_dir, int(cfg["size"]))

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=int(cfg["radius"]),
        neighbors=int(cfg["neighbors"]),
        grid_x=int(cfg["grid_x"]),
        grid_y=int(cfg["grid_y"]),
    )
    recognizer.setThreshold(float(cfg["threshold"]))
    recognizer.train(images, np.array(labels))

    model_path = Path(cfg["model_path"])
    labels_path = Path(cfg["labels_path"])
    ensure_dir(model_path.parent)
    recognizer.write(str(model_path))
    save_label_map(label_map, labels_path)

    print(f"Model saved to {model_path}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
