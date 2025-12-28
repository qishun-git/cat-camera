from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from cat_face.detection import create_detector
from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer
from cat_face.utils import ensure_dir, load_label_map, load_project_config, preprocess_face, resolve_paths


def iterate_clips(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.mp4"))


def move_unique(src: Path, dest_dir: Path) -> Path:
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.rename(dest)
    return dest


def load_recognizer(config: Dict[str, object], paths: Dict[str, Path]) -> Tuple[Optional[EmbeddingRecognizer], Dict[int, str]]:
    training_cfg = config.get("training", {})
    recog_cfg = config.get("recognition", {})
    models_dir = paths["models"]
    model_path = models_dir / recog_cfg.get("embedding_model_filename", training_cfg.get("embedding_model_filename", "embeddings.npz"))
    labels_path = models_dir / recog_cfg.get("labels_filename", training_cfg.get("labels_filename", "labels.json"))
    try:
        labels = load_label_map(labels_path)
    except FileNotFoundError:
        return None, {}
    try:
        embedding_model = EmbeddingModel.load(model_path)
    except FileNotFoundError:
        return None, {}
    extractor = EmbeddingExtractor(input_size=int(training_cfg.get("embedding_input_size", 224)))
    recognizer = EmbeddingRecognizer(
        model=embedding_model,
        extractor=extractor,
        threshold=float(recog_cfg.get("embedding_threshold", 0.75)),
    )
    return recognizer, labels


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"face_size": 100} | config.get("vision", {})
    recorder_cfg = config.get("recorder", {})
    processing_cfg: Dict[str, object] = {
        "clips_dir": recorder_cfg.get("output_dir"),
        "save_limit": None,
        "training_refresh_count": 10,
        "recognition_margin": 0.05,
    } | config.get("clip_processing", {})

    clips_dir = Path(processing_cfg["clips_dir"] or (paths["base"] / "clips"))
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    detector = create_detector(config.get("detection"))
    recognizer, labels_map = load_recognizer(config, paths)
    face_size = int(vision_cfg["face_size"])
    recognized_clips_root = ensure_dir(paths["base"] / "recognized_clips")
    unknown_clips_root = ensure_dir(paths["base"] / "unknown_clips")

    save_limit = int(processing_cfg.get("save_limit") or 0)
    training_refresh_count = int(processing_cfg.get("training_refresh_count", 10))
    recognition_margin = float(processing_cfg.get("recognition_margin", 0.05))

    total_saved = 0
    clip_paths = iterate_clips(clips_dir)
    if not clip_paths:
        print(f"No clips found in {clips_dir}")
        return

    for clip_path in clip_paths:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Warning: unable to open {clip_path}, skipping.")
            continue

        detection_samples: List[Tuple[int, int, Any]] = []
        recognized_samples: List[Tuple[str, float, Any]] = []
        best_label: Optional[str] = None
        best_score = float("-inf")
        frame_index = 0
        print(f"Processing clip: {clip_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            faces = detector.detect(frame)
            for idx, (x, y, w, h) in enumerate(faces):
                crop = frame[y : y + h, x : x + w]
                processed = preprocess_face(crop, size=(face_size, face_size))
                detection_samples.append((frame_index, idx, processed))
                if recognizer:
                    label_id, score = recognizer.predict(processed)
                    if label_id != -1:
                        label_name = labels_map.get(label_id)
                        if label_name:
                            recognized_samples.append((label_name, score, processed))
                            if score >= recognizer.threshold + recognition_margin and score > best_score:
                                best_label = label_name
                                best_score = score
        cap.release()

        saved_for_clip = 0
        if best_label:
            # Promote training samples
            candidates = [img for (label, score, img) in recognized_samples if label == best_label]
            if not candidates:
                candidates = [img for (_, _, img) in recognized_samples]
            if candidates:
                if len(candidates) > training_refresh_count > 0:
                    candidates = random.sample(candidates, training_refresh_count)
                target_dir = ensure_dir(paths["training"] / best_label)
                for idx, processed in enumerate(candidates):
                    filename = target_dir / f"{clip_path.stem}_auto_{idx}.png"
                    cv2.imwrite(str(filename), processed)
                print(f"Promoted {len(candidates)} frame(s) to training/{best_label}.")
            move_unique(clip_path, recognized_clips_root / best_label)
        else:
            clip_folder = ensure_dir(paths["unlabeled"] / clip_path.stem)
            samples = detection_samples
            if save_limit > 0 and len(samples) > save_limit:
                samples = random.sample(samples, save_limit)
            for frame_idx, det_idx, processed in samples:
                filename = clip_folder / f"f{frame_idx}_n{det_idx}.png"
                cv2.imwrite(str(filename), processed)
                saved_for_clip += 1
            if samples:
                total_saved += saved_for_clip
                print(f"Saved {saved_for_clip} face(s) from {clip_path} into {clip_folder}.")
            else:
                print(f"No faces extracted from {clip_path}.")
            move_unique(clip_path, unknown_clips_root)

    print(f"Done. Total unlabeled faces saved: {total_saved}")


if __name__ == "__main__":
    main()
