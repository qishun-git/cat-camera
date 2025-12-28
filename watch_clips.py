from __future__ import annotations

import subprocess
import time
from pathlib import Path

from process_clips import main as process_clips_main
from cat_face.utils import ensure_dir, load_project_config, resolve_paths


def compress_clip(src: Path, dest_dir: Path, crf: int) -> None:
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
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
    subprocess.run(cmd, check=True)
    src.unlink(missing_ok=True)
    print(f"Compressed {src} -> {dest}")


def main() -> None:
    config = load_project_config()
    paths = resolve_paths(config)
    recorder_cfg = config.get("recorder", {})
    processing_cfg = config.get("clip_processing", {})
    raw_dir = Path(recorder_cfg.get("output_dir") or (paths["base"] / "clips"))
    compressed_dir = Path(processing_cfg.get("clips_dir") or (paths["base"] / "compressed_clips"))
    crf = int(processing_cfg.get("compression_crf", 28))
    interval = float(processing_cfg.get("watch_interval", 5.0))

    if not raw_dir.exists():
        raise FileNotFoundError(f"Recorder output directory not found: {raw_dir}")
    ensure_dir(compressed_dir)

    print(f"Watching {raw_dir} for new clips. Compressed output -> {compressed_dir}")
    try:
        while True:
            new_clips = sorted(p for p in raw_dir.glob("*.mp4") if not p.stem.endswith("_tmp"))
            for clip in new_clips:
                try:
                    compress_clip(clip, compressed_dir, crf)
                except subprocess.CalledProcessError as exc:
                    print(f"Compression failed for {clip}: {exc}")
            process_clips_main()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Watcher stopped.")


if __name__ == "__main__":
    main()
