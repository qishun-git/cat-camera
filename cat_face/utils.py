from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml

# Default locations that keep training artifacts tidy.
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
CONFIG_DIR = Path("configs")
PROJECT_CONFIG_FILE = CONFIG_DIR / "cat_face.yaml"


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def preprocess_face(image: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """Resize a face crop to the desired size, preserving color information."""
    if image is None:
        raise ValueError("Cannot preprocess an empty image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    face = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return face.astype("uint8")


def save_label_map(mapping: Dict[int, str], path: Path | str) -> None:
    """Persist label-id to class-name mapping."""
    path = Path(path)
    ensure_dir(path.parent)
    serializable = {str(idx): name for idx, name in sorted(mapping.items())}
    path.write_text(json.dumps(serializable, indent=2))


def load_label_map(path: Path | str) -> Dict[int, str]:
    """Load label map JSON and convert ids to ints."""
    data = json.loads(Path(path).read_text())
    return {int(idx): name for idx, name in data.items()}


def iter_image_files(root: Path, extensions: Iterable[str] = (".jpg", ".jpeg", ".png")) -> Iterable[Path]:
    """Yield image files in a directory."""
    for path in sorted(root.glob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            yield path


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Read a YAML file and return the parsed payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return data


def list_images_sorted_by_age(directory: Path) -> List[Path]:
    """Return image files sorted by modification time ascending (oldest first)."""
    files = list(iter_image_files(directory))
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def rotate_files(directory: Path, limit: int) -> int:
    """Delete oldest files until the directory holds <= limit images."""
    if limit <= 0 or not directory.exists():
        return 0
    files = list_images_sorted_by_age(directory)
    removed = 0
    while len(files) > limit:
        oldest = files.pop(0)
        try:
            oldest.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: failed to delete {oldest}: {exc}")
    return removed


def _require(config: Dict[str, Any], key_path: List[str]) -> None:
    cursor: Any = config
    for idx, key in enumerate(key_path):
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(
                f"Missing required config key: {'.'.join(key_path[: idx + 1])}"
            )
        cursor = cursor[key]


def load_project_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load the unified project configuration and ensure required keys exist."""
    config_path = Path(path) if path else PROJECT_CONFIG_FILE
    cfg = load_yaml(config_path)
    # Required keys: detection.model and streaming.public_url for the web UI.
    _require(cfg, ["detection", "model"])
    _require(cfg, ["streaming", "public_url"])
    return cfg


def _expand_path(value: str | Path | None, default: Path) -> Path:
    if value:
        return Path(value).expanduser()
    return default


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve common paths defined in the project config."""
    paths_cfg = config.get("paths", {})
    base = _expand_path(paths_cfg.get("base_data_dir"), DATA_DIR)
    unlabeled_default = base / "unlabeled"
    training_default = base / "training"
    reject_default = base / "rejected"
    models_default = _expand_path(paths_cfg.get("models_dir"), MODEL_DIR)
    resolved = {
        "base": base,
        "unlabeled": _expand_path(paths_cfg.get("unlabeled_dir"), unlabeled_default),
        "training": _expand_path(paths_cfg.get("training_dir"), training_default),
        "reject": _expand_path(paths_cfg.get("reject_dir"), reject_default),
        "models": models_default,
    }
    return resolved
