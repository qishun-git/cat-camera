from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2

from cat_face.utils import CONFIG_DIR, DATA_DIR, ensure_dir, load_yaml

CONFIG_PATH = CONFIG_DIR / "sort.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    cfg = load_yaml(path)
    cfg.setdefault("unlabeled_root", str(DATA_DIR / "unlabeled"))
    cfg.setdefault("destination_root", str(DATA_DIR))
    cfg.setdefault("delete_rejects", False)
    cfg.setdefault("reject_folder", "rejected")
    cfg.setdefault("window_name", "Sort Unlabeled")
    cfg.setdefault("image_extensions", [".png", ".jpg", ".jpeg"])
    cfg.setdefault("window_width", 640)
    cfg.setdefault("window_height", 480)
    cfg.setdefault("window_x", 100)
    cfg.setdefault("window_y", 100)
    return cfg


def gather_images(root: Path, exts: List[str]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Unlabeled directory not found: {root}")
    files: List[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in exts and path.is_file():
            files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def discover_labels(dest_root: Path, unlabeled_root: Path, reject_folder: str) -> List[str]:
    if not dest_root.exists():
        return []
    labels: List[str] = []
    reject_path = (dest_root / reject_folder).resolve()
    try:
        unlabeled_resolved = unlabeled_root.resolve()
    except FileNotFoundError:
        unlabeled_resolved = None
    for path in sorted(dest_root.iterdir()):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved == reject_path or (unlabeled_resolved and resolved == unlabeled_resolved):
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


def reject_image(src: Path, dest_root: Path, reject_folder: str, delete_rejects: bool) -> None:
    if delete_rejects:
        src.unlink(missing_ok=True)
        print(f"Deleted {src}")
        return
    reject_dir = ensure_dir(dest_root / reject_folder)
    target_path = reject_dir / src.name
    counter = 1
    while target_path.exists():
        target_path = reject_dir / f"{src.stem}_{counter}{src.suffix}"
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


def main() -> None:
    cfg = load_config()
    unlabeled_root = Path(cfg["unlabeled_root"])
    destination_root = Path(cfg["destination_root"])
    known_labels = discover_labels(destination_root, unlabeled_root, cfg["reject_folder"])
    if known_labels:
        print(f"Discovered labels: {', '.join(known_labels)}")
    else:
        print("No label folders detected; you will need to type label names manually.")

    files = gather_images(unlabeled_root, [ext.lower() for ext in cfg["image_extensions"]])
    if not files:
        print(f"No images found in {unlabeled_root}")
        return

    window_name = cfg["window_name"]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg["window_width"]), int(cfg["window_height"]))
    cv2.moveWindow(window_name, int(cfg["window_x"]), int(cfg["window_y"]))

    keymap = {str(idx + 1): label for idx, label in enumerate(known_labels)}
    print("Sorting session started.")
    print("Commands: enter label name, digit shortcut, 'skip' to leave, 'delete'/'d' to discard, 'q' to quit.")

    for path in files:
        source_parent = path.parent
        image = cv2.imread(str(path))
        if image is None:
            print(f"Warning: unable to read {path}, skipping.")
            continue
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

        while True:
            if known_labels:
                shortcuts = ", ".join(f"{digit}->{label}" for digit, label in keymap.items())
                prompt = f"{path} | shortcuts [{shortcuts}] > "
            else:
                prompt = f"{path} > "
            user_input = input(prompt).strip()
            lower_input = user_input.lower()
            if not user_input or lower_input == "skip":
                print(f"Skipped {path}")
                break
            if lower_input == "q":
                print("Exiting sorter.")
                cv2.destroyAllWindows()
                return
            if lower_input in {"delete", "d"}:
                reject_image(path, destination_root, cfg["reject_folder"], cfg["delete_rejects"])
                cleanup_empty_dirs(source_parent, unlabeled_root)
                break
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
    print("Sorting session complete.")


if __name__ == "__main__":
    main()
