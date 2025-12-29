from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2

from cat_face.utils import configure_logging, ensure_dir, load_project_config, resolve_paths

logger = logging.getLogger(__name__)


def find_clips_with_annotations(root: Path) -> List[Path]:
    return sorted(
        clip
        for clip in root.rglob("*.mp4")
        if clip.with_suffix(f"{clip.suffix}.json").exists()
    )


def load_annotations(sidecar_path: Path) -> Dict[str, object]:
    with sidecar_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_annotations(frame, overlays, highlight_label: str) -> None:
    for entry in overlays:
        bbox = entry.get("bbox", [0, 0, 0, 0])
        x, y, w, h = map(int, bbox)
        label = entry.get("label")
        score = entry.get("score", 0.0)
        if label == highlight_label:
            color = (0, 255, 0)
        elif label:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        text = label or "unknown"
        if label:
            text = f"{text} ({score:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def play_clip(clip_path: Path, annot_data: Dict[str, object]) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("Unable to open clip: %s", clip_path)
        return
    highlight_label = annot_data.get("highlight_label")
    frames = annot_data.get("frames", {})

    logger.info("Press Q to exit playback.")
    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            overlays = frames.get(str(frame_index), [])
            draw_annotations(frame, overlays, highlight_label)
            cv2.imshow(f"Clip: {clip_path.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def choose_clip(clips: List[Path]) -> Path | None:
    if not clips:
        return None
    while True:
        logger.info("Available annotated clips:")
        for idx, clip in enumerate(clips, start=1):
            label = clip.parent.name
            logger.info("  %s. [%s] %s", idx, label, clip.name)
        logger.info("  q. Quit")
        choice = input("Select a clip to view: ").strip().lower()
        if choice in {"q", "quit", ""}:
            return None
        if not choice.isdigit():
            logger.info("Please enter a number or 'q' to quit.")
            continue
        index = int(choice)
        if 1 <= index <= len(clips):
            return clips[index - 1]
        logger.info("Invalid selection. Try again.")


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    recognized_root = ensure_dir(paths["base"] / "recognized_clips")
    clips = find_clips_with_annotations(recognized_root)
    if not clips:
        logger.info("No annotated clips found under %s.", recognized_root)
        return

    while True:
        clip = choose_clip(clips)
        if clip is None:
            logger.info("Viewer exited.")
            return
        sidecar = clip.with_suffix(f"{clip.suffix}.json")
        try:
            data = load_annotations(sidecar)
        except FileNotFoundError:
            logger.warning("Sidecar missing for %s: %s", clip, sidecar)
            continue
        play_clip(clip, data)


if __name__ == "__main__":
    configure_logging()
    main()
