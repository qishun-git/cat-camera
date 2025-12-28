from __future__ import annotations

import random
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


def split_dataset(
    images: List[np.ndarray],
    labels: List[int],
    validation_split: float,
    rng_seed: int = 42,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    if validation_split <= 0:
        return images, labels, [], []

    per_class: Dict[int, List[np.ndarray]] = {}
    for img, label in zip(images, labels):
        per_class.setdefault(label, []).append(img)

    rng = random.Random(rng_seed)
    train_images: List[np.ndarray] = []
    train_labels: List[int] = []
    val_images: List[np.ndarray] = []
    val_labels: List[int] = []

    for label, samples in per_class.items():
        rng.shuffle(samples)
        total = len(samples)
        if total < 2:
            train_images.extend(samples)
            train_labels.extend([label] * total)
            continue
        val_count = int(round(total * validation_split))
        if val_count <= 0:
            val_count = 1
        if val_count >= total:
            val_count = total - 1
        val_split = samples[:val_count]
        train_split = samples[val_count:]
        val_images.extend(val_split)
        val_labels.extend([label] * len(val_split))
        train_images.extend(train_split)
        train_labels.extend([label] * len(train_split))

    return train_images, train_labels, val_images, val_labels


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_defaults = {
        "face_size": 100,
    }
    vision_cfg = vision_defaults | config.get("vision", {})
    defaults = {
        "radius": 2,
        "neighbors": 8,
        "grid_x": 8,
        "grid_y": 8,
        "threshold": 80.0,
        "model_filename": "lbph_model.xml",
        "labels_filename": "labels.json",
        "validation_split": 0.0,
    }
    train_cfg = defaults | config.get("training", {})

    data_dir = Path(train_cfg.get("data_dir", paths["training"]))
    face_size = int(vision_cfg["face_size"])
    images, labels, label_map = load_dataset(data_dir, face_size)
    (
        train_images,
        train_labels,
        val_images,
        val_labels,
    ) = split_dataset(images, labels, float(train_cfg["validation_split"]))
    if len(train_images) < 2:
        print("Warning: validation split left too few samples; training on full dataset without validation.")
        train_images, train_labels = images, labels
        val_images, val_labels = [], []

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=int(train_cfg["radius"]),
        neighbors=int(train_cfg["neighbors"]),
        grid_x=int(train_cfg["grid_x"]),
        grid_y=int(train_cfg["grid_y"]),
    )
    recognizer.setThreshold(float(train_cfg["threshold"]))
    recognizer.train(train_images, np.array(train_labels))

    models_dir = ensure_dir(paths["models"])
    model_path = models_dir / train_cfg["model_filename"]
    labels_path = models_dir / train_cfg["labels_filename"]
    recognizer.write(str(model_path))
    save_label_map(label_map, labels_path)

    print(f"Model saved to {model_path}")
    print(f"Labels saved to {labels_path}")
    if val_images:
        correct = 0
        confidences: List[float] = []
        for img, label in zip(val_images, val_labels):
            pred_label, confidence = recognizer.predict(img)
            if np.isfinite(confidence):
                confidences.append(confidence)
            if pred_label == label:
                correct += 1
        total = len(val_labels)
        accuracy = (correct / total) * 100
        mean_conf = float(np.mean(confidences)) if confidences else float("nan")
        print(f"Validation samples: {total}")
        print(f"Validation accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"Average LBPH confidence: {mean_conf:.2f}")


if __name__ == "__main__":
    main()
