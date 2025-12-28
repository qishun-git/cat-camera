from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer
from cat_face.utils import ensure_dir, load_label_map, load_project_config, preprocess_face, resolve_paths

SORT_DEFAULTS: Dict[str, object] = {
    "window_name": "Sort Unlabeled",
    "window_width": 640,
    "window_height": 480,
    "window_x": 100,
    "window_y": 100,
    "delete_rejects": False,
    "image_extensions": [".png", ".jpg", ".jpeg"],
}


def gather_images(root: Path, exts: List[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Unlabeled directory not found: {root}")
    files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in exts and path.is_file():
            files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def discover_labels(dest_root: Path, unlabeled_root: Path, reject_path: Path) -> List[str]:
    if not dest_root.exists():
        return []
    labels: List[str] = []
    try:
        unlabeled_resolved = unlabeled_root.resolve()
    except FileNotFoundError:
        unlabeled_resolved = None
    try:
        reject_resolved = reject_path.resolve()
    except FileNotFoundError:
        reject_resolved = None
    for path in sorted(dest_root.iterdir()):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if (reject_resolved and resolved == reject_resolved) or (unlabeled_resolved and resolved == unlabeled_resolved):
            continue
        labels.append(path.name)
    return labels


def move_to_label(src: Path, dest_root: Path, label: str) -> None:
    target_dir = ensure_dir(dest_root / label)
    target_path = target_dir / src.name
    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.rename(target_path)
    print(f"Moved {src} -> {target_path}")


def reject_image(src: Path, reject_dir: Path, delete_rejects: bool) -> None:
    if delete_rejects:
        src.unlink(missing_ok=True)
        print(f"Deleted {src}")
        return
    target_dir = ensure_dir(reject_dir)
    target_path = target_dir / src.name
    counter = 1
    while target_path.exists():
        target_path = target_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.rename(target_path)
    print(f"Moved {src} -> {target_path}")


def cleanup_empty_dirs(start: Path, stop_at: Path) -> None:
    """Remove empty directories between start and stop_at (exclusive)."""
    stop = stop_at.resolve()
    current = start
    while current.exists() and current.resolve() != stop:
        try:
            if any(current.iterdir()):
                break
            current.rmdir()
        except OSError:
            break
        current = current.parent
    # Optionally remove stop itself if empty
    if current.exists() and current.resolve() == stop:
        try:
            if not any(current.iterdir()):
                current.rmdir()
        except OSError:
            pass


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


def predict_label(
    image: Any,
    recognizer: Optional[EmbeddingRecognizer],
    labels: Dict[int, str],
    face_size: int,
) -> Optional[str]:
    if recognizer is None:
        return None
    processed = preprocess_face(image, size=(face_size, face_size))
    label_id, score = recognizer.predict(processed)
    if label_id == -1:
        return None
    name = labels.get(label_id)
    if not name:
        return None
    return f"{name} ({score:.2f})"


def move_folder_to_label(folder: Path, destination_root: Path, label: str) -> None:
    images = sorted(p for p in folder.glob("*") if p.is_file())
    for path in images:
        move_to_label(path, destination_root, label)
    print(f"Moved entire folder {folder} -> label '{label}'")


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    sort_cfg = SORT_DEFAULTS | config.get("sorter", {})
    vision_cfg = {"face_size": 100} | config.get("vision", {})

    unlabeled_root = ensure_dir(paths["unlabeled"])
    destination_root = ensure_dir(paths["training"])
    reject_dir = ensure_dir(paths["reject"])

    known_labels = discover_labels(destination_root, unlabeled_root, reject_dir)
    if known_labels:
        print(f"Discovered labels: {', '.join(known_labels)}")
    else:
        print("No label folders detected; you will need to type label names manually.")

    files = gather_images(unlabeled_root, [ext.lower() for ext in sort_cfg["image_extensions"]])
    if not files:
        print(f"No images found in {unlabeled_root}")
        return

    window_name = sort_cfg["window_name"]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(sort_cfg["window_width"]), int(sort_cfg["window_height"]))
    cv2.moveWindow(window_name, int(sort_cfg["window_x"]), int(sort_cfg["window_y"]))

    keymap = {str(idx + 1): label for idx, label in enumerate(known_labels)}
    recognizer, labels_map = load_recognizer(config, paths)
    print("Sorting session started.")
    print("Commands: enter label name, digit shortcut, 'folder <label>' to label entire folder, 'skip' to leave, 'delete'/'d' to discard, 'q' to quit.")

    for path in files:
        source_parent = path.parent
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Warning: unable to read {path}, skipping.")
            continue
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        suggestion = predict_label(image, recognizer, labels_map, int(vision_cfg["face_size"]))

        while True:
            if known_labels:
                shortcuts = ", ".join(f"{digit}->{label}" for digit, label in keymap.items())
                prompt = f"{path} | shortcuts [{shortcuts}]"
            else:
                prompt = f"{path}"
            if suggestion:
                prompt += f" | suggest {suggestion}"
            prompt += " > "
            user_input = input(prompt).strip()
            lower_input = user_input.lower()
            selected_label: Optional[str] = None
            if not user_input:
                if suggestion:
                    selected_label = suggestion.split(" (", 1)[0]
                else:
                    print(f"Skipped {path}")
                    break
            if not selected_label and lower_input == "skip":
                print(f"Skipped {path}")
                break
            if not selected_label and lower_input == "q":
                print("Exiting sorter.")
                cv2.destroyAllWindows()
                return
            if not selected_label and lower_input in {"delete", "d"}:
                reject_image(path, reject_dir, bool(sort_cfg["delete_rejects"]))
                cleanup_empty_dirs(source_parent, unlabeled_root)
                break
            if not selected_label and lower_input.startswith("folder"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].strip():
                    print("Usage: folder <label>")
                    continue
                folder_label = parts[1].strip()
                move_folder_to_label(source_parent, destination_root, folder_label)
                if folder_label not in known_labels:
                    known_labels.append(folder_label)
                    known_labels.sort()
                    keymap = {str(idx + 1): lbl for idx, lbl in enumerate(known_labels)}
                    print(f"Updated shortcuts: {', '.join(f'{k}->{v}' for k, v in keymap.items())}")
                cleanup_empty_dirs(source_parent, unlabeled_root)
                break
            if selected_label:
                label = selected_label
            else:
                label = keymap.get(user_input, user_input)
            if not label:
                print("Please provide a non-empty label.")
                continue
            move_to_label(path, destination_root, label)
            if label not in known_labels:
                known_labels.append(label)
                known_labels.sort()
                keymap = {str(idx + 1): lbl for idx, lbl in enumerate(known_labels)}
                print(f"Updated shortcuts: {', '.join(f'{k}->{v}' for k, v in keymap.items())}")
            cleanup_empty_dirs(source_parent, unlabeled_root)
            break

    cv2.destroyAllWindows()
    per_label_limit = int(sort_cfg.get("per_label_limit", 0))
    if per_label_limit > 0:
        print(f"Applying per-label cap of {per_label_limit} image(s).")
        import random

        for label_dir in destination_root.iterdir():
            if not label_dir.is_dir():
                continue
            images = list(label_dir.glob("*"))
            if len(images) <= per_label_limit:
                continue
            keep = set(random.sample(images, per_label_limit))
            removed = 0
            for img in images:
                if img not in keep:
                    img.unlink(missing_ok=True)
                    removed += 1
            print(f"Capped {label_dir.name}: removed {removed} image(s).")
    print("Sorting session complete.")


if __name__ == "__main__":
    main()
