from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import json

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


def write_annotation_sidecar(
    clip_path: Path,
    highlight_label: str,
    detection_total: int,
    recognized_total: int,
    recognized_majority_needed: int,
    detection_majority_needed: int,
    counts: Dict[str, int],
    label_best_score: Dict[str, float],
    frame_annotations: Dict[int, List[Tuple[Tuple[int, int, int, int], Optional[str], float]]],
) -> None:
    json_path = clip_path.with_suffix(f"{clip_path.suffix}.json")
    frames_payload = {
        str(idx): [
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "label": label_name,
                "score": float(score),
            }
            for ((x, y, w, h), label_name, score) in entries
        ]
        for idx, entries in sorted(frame_annotations.items())
    }
    payload = {
        "clip": clip_path.name,
        "highlight_label": highlight_label,
        "detections_total": detection_total,
        "recognized_total": recognized_total,
        "recognized_majority_needed": recognized_majority_needed,
        "detection_majority_needed": detection_majority_needed,
        "label_counts": counts,
        "label_best_scores": label_best_score,
        "frames": frames_payload,
    }
    json_path.write_text(json.dumps(payload))
    print(f"Wrote annotation sidecar: {json_path}")


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
        "detection_interval": recorder_cfg.get("detection_interval", 0.5),
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

    detection_interval = float(processing_cfg.get("detection_interval", 0.5) or 0.0)

    for clip_path in clip_paths:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Warning: unable to open {clip_path}, skipping.")
            continue

        detection_samples: List[Tuple[int, int, Any]] = []
        recognized_samples: List[Tuple[str, float, Any]] = []
        frame_annotations: Dict[int, List[Tuple[Tuple[int, int, int, int], Optional[str], float]]] = {}
        frame_index = 0
        print(f"Processing clip: {clip_path}")
        last_detection_ts = -float("inf")
        last_detections: List[Tuple[int, int, int, int]] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            now = frame_index / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if detection_interval <= 0 or (now - last_detection_ts) >= detection_interval:
                faces = detector.detect(frame)
                last_detection_ts = now
                last_detections = faces
            else:
                faces = last_detections
            for idx, (x, y, w, h) in enumerate(faces):
                crop = frame[y : y + h, x : x + w]
                processed = preprocess_face(crop, size=(face_size, face_size))
                detection_samples.append((frame_index, idx, processed))
                label_name: Optional[str] = None
                score_value = 0.0
                if recognizer:
                    label_id, score_value = recognizer.predict(processed)
                    if label_id != -1:
                        label_name = labels_map.get(label_id)
                        if label_name:
                            recognized_samples.append((label_name, score_value, processed))
                frame_annotations.setdefault(frame_index, []).append(((x, y, w, h), label_name, float(score_value)))
        cap.release()

        best_label: Optional[str] = None
        best_score = float("-inf")
        counts: Dict[str, int] = {}
        label_best_score: Dict[str, float] = {}
        recognized_total = len(recognized_samples)
        detection_total = len(detection_samples)
        recognized_majority_needed = recognized_total // 2 + 1 if recognized_total else 0
        detection_majority_needed = detection_total // 2 + 1 if detection_total else 0
        if recognizer and recognized_samples:
            for label_name, score, _ in recognized_samples:
                counts[label_name] = counts.get(label_name, 0) + 1
                label_best_score[label_name] = max(score, label_best_score.get(label_name, float("-inf")))
            for label_name, count in counts.items():
                if count < recognized_majority_needed:
                    continue
                if detection_total and count < detection_majority_needed:
                    continue
                score = label_best_score[label_name]
                if score >= recognizer.threshold + recognition_margin and score > best_score:
                    best_label = label_name
                    best_score = score

        summary_prefix = (
            f"[SUMMARY] {clip_path.name}: detections={detection_total}, recognized={recognized_total}"
        )
        if best_label:
            label_count = counts.get(best_label, 0)
            print(
                f"{summary_prefix} -> auto-labeled '{best_label}' "
                f"(label detections {label_count}/{detection_total})"
            )
        else:
            if detection_total == 0:
                reason = "no faces detected"
            elif not recognizer:
                reason = "recognizer unavailable"
            elif recognized_total == 0:
                reason = "faces detected but none matched known cats"
            else:
                top_label = max(counts, key=counts.get) if counts else None
                if top_label is None:
                    reason = "no recognized label counts available"
                else:
                    top_count = counts[top_label]
                    if top_count < recognized_majority_needed:
                        reason = (
                            f"no label reached majority of recognized faces "
                            f"(top '{top_label}' {top_count}/{recognized_total})"
                        )
                    elif detection_total and top_count < detection_majority_needed:
                        reason = (
                            f"top label '{top_label}' did not cover majority of detections "
                            f"({top_count}/{detection_total})"
                        )
                    else:
                        margin = recognizer.threshold + recognition_margin
                        score = label_best_score.get(top_label, float("-inf"))
                        reason = f"confidence {score:.2f} below required {margin:.2f} for '{top_label}'"
            print(f"{summary_prefix} -> left unlabeled ({reason})")

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
            recognized_clip_path = move_unique(clip_path, recognized_clips_root / best_label)
            try:
                write_annotation_sidecar(
                    recognized_clip_path,
                    best_label,
                    detection_total,
                    recognized_total,
                    recognized_majority_needed,
                    detection_majority_needed,
                    counts,
                    label_best_score,
                    frame_annotations,
                )
            except Exception as exc:
                print(f"Warning: failed to write annotations for {recognized_clip_path}: {exc}")
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
