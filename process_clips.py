from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2

from cat_face.detection import create_detector
from cat_face.utils import ensure_dir, load_project_config, preprocess_face, resolve_paths


def iterate_clips(directory: Path) -> Path:
    for path in sorted(directory.glob("*.mp4")):
        yield path


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    vision_cfg = {"face_size": 100} | config.get("vision", {})

    recorder_cfg = config.get("recorder", {})
    processing_cfg: Dict[str, object] = {
        "clips_dir": recorder_cfg.get("output_dir"),
        "mode": "unlabeled",
        "cat_name": recorder_cfg.get("cat_name"),
        "save_limit": None,
    } | config.get("clip_processing", {})

    clips_dir = Path(processing_cfg["clips_dir"] or (paths["base"] / "clips"))
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    detector = create_detector(config.get("detection"))
    face_size = int(vision_cfg["face_size"])

    save_limit = processing_cfg.get("save_limit")
    if save_limit is not None:
        save_limit = int(save_limit)

    total_saved = 0
    for clip_path in iterate_clips(clips_dir):
        clip_folder = ensure_dir(paths["unlabeled"] / clip_path.stem)
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Warning: unable to open {clip_path}, skipping.")
            continue

        queued_paths: list[Path] = []
        saved_for_clip = 0
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
                filename = clip_folder / f"f{frame_index}_n{idx}.png"
                cv2.imwrite(str(filename), processed)
                if save_limit:
                    queued_paths.append(filename)
                saved_for_clip += 1

        cap.release()
        if save_limit and queued_paths:
            import random

            if len(queued_paths) > save_limit:
                selected = set(random.sample(queued_paths, save_limit))
                for path in queued_paths:
                    if path not in selected:
                        path.unlink(missing_ok=True)
                saved_for_clip = save_limit
            else:
                saved_for_clip = len(queued_paths)

        total_saved += saved_for_clip
        print(f"Saved {saved_for_clip} faces from {clip_path}")

    print(f"Done. Total faces saved: {total_saved}")


if __name__ == "__main__":
    main()
