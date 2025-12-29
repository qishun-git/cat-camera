from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from process_clips import main as process_clips_main
from cat_face.utils import configure_logging, ensure_dir, load_project_config, resolve_paths

logger = logging.getLogger(__name__)


def _unique_destination(dest_dir: Path, src: Path) -> Path:
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    return dest


def move_clip(src: Path, dest_dir: Path) -> None:
    dest = _unique_destination(dest_dir, src)
    src.replace(dest)
    clip_recording_marker(src).unlink(missing_ok=True)
    logger.info("Moved %s -> %s (compression disabled)", src, dest)


def compress_clip(src: Path, dest_dir: Path, crf: int) -> None:
    dest = _unique_destination(dest_dir, src)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx265",
        "-tag:v",
        "hvc1",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    src.unlink(missing_ok=True)
    clip_recording_marker(src).unlink(missing_ok=True)
    logger.info("Compressed %s -> %s", src, dest)


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    recorder_cfg = config.get("recorder", {})
    processing_cfg = config.get("clip_processing", {})
    raw_dir = Path(recorder_cfg.get("output_dir") or (paths["base"] / "clips"))
    compressed_dir = Path(processing_cfg.get("clips_dir") or (paths["base"] / "compressed_clips"))
    crf = int(processing_cfg.get("compression_crf", 28))
    interval = float(processing_cfg.get("watch_interval", 5.0))
    enable_compression = bool(processing_cfg.get("enable_compression", True))

    if not raw_dir.exists():
        raise FileNotFoundError(f"Recorder output directory not found: {raw_dir}")
    ensure_dir(compressed_dir)

    logger.info(
        "Watching %s for new clips. Output -> %s (%s)",
        raw_dir,
        compressed_dir,
        "compressing" if enable_compression else "copy-only",
    )
    try:
        while True:
            new_clips = sorted(p for p in raw_dir.glob("*.mp4") if not clip_recording_marker(p).exists())
            for clip in new_clips:
                try:
                    if enable_compression:
                        compress_clip(clip, compressed_dir, crf)
                    else:
                        move_clip(clip, compressed_dir)
                except subprocess.CalledProcessError as exc:
                    logger.error("Compression failed for %s: %s", clip, exc)
            process_clips_main()
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Watcher stopped.")


if __name__ == "__main__":
    configure_logging()
    main()
def clip_recording_marker(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.recording")
