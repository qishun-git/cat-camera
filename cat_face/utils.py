from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np
import yaml

# Default locations that keep training artifacts tidy.
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
CONFIG_DIR = Path("configs")


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_cascade_path(filename: str = "haarcascade_frontalcatface_extended.xml") -> Path:
    """Return the bundled OpenCV cascade path for cat faces."""
    cascade_dir = Path(cv2.data.haarcascades)
    return cascade_dir / filename


def preprocess_face(image: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """Convert an image to grayscale, equalize, resize, and return uint8 array."""
    if image is None:
        raise ValueError("Cannot preprocess an empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(image)
    face = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)
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
