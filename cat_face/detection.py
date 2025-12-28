from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cat_face.utils import default_cascade_path

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None


Box = Tuple[int, int, int, int]


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[Box]:
        raise NotImplementedError


@dataclass
class CascadeParams:
    cascade_path: Path
    scale_factor: float
    min_neighbors: int
    min_size: int


class CascadeDetector(BaseDetector):
    def __init__(self, params: CascadeParams) -> None:
        if not params.cascade_path.exists():
            raise FileNotFoundError(f"Cascade file not found: {params.cascade_path}")
        self.params = params
        self.classifier = cv2.CascadeClassifier(str(params.cascade_path))

    def detect(self, frame: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.params.scale_factor,
            minNeighbors=self.params.min_neighbors,
            minSize=(self.params.min_size, self.params.min_size),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


class YoloOnnxDetector(BaseDetector):
    def __init__(
        self,
        model_path: Path,
        input_size: Tuple[int, int],
        class_ids: Optional[Sequence[int]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if ort is None:  # pragma: no cover - import guard
            raise RuntimeError("YOLO detector requires onnxruntime; install it via pip.")
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.input_size = input_size
        self.class_ids = set(int(cid) for cid in class_ids) if class_ids else None
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        sess_opts = ort.SessionOptions()
        session_providers = list(providers) if providers else None
        self.session = ort.InferenceSession(str(model_path), sess_opts, providers=session_providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> List[Box]:
        processed, ratio, pad = self._prepare_input(frame)
        outputs = self.session.run(None, {self.input_name: processed})
        predictions = outputs[0]
        if predictions.ndim == 3:
            predictions = predictions[0]
        boxes: List[Box] = []
        confidences: List[float] = []
        for det in predictions:
            obj_conf = det[4]
            if obj_conf < 1e-5:
                continue
            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            if self.class_ids is not None and class_id not in self.class_ids:
                continue
            score = float(obj_conf * class_scores[class_id])
            if score < self.conf_threshold:
                continue
            box = self._xywh_to_x1y1x2y2(det[:4], ratio, pad, frame.shape[1], frame.shape[0])
            boxes.append(box)
            confidences.append(score)

        keep_indices = self._nms(boxes, confidences, self.iou_threshold)
        final_boxes = [boxes[i] for i in keep_indices]
        return final_boxes

    def _prepare_input(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        top = int(math.floor(dh))
        left = int(math.floor(dw))
        padded[top : top + new_h, left : left + new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, (dw, dh)

    def _xywh_to_x1y1x2y2(
        self,
        box: np.ndarray,
        scale: float,
        pad: Tuple[float, float],
        image_w: int,
        image_h: int,
    ) -> Box:
        x_c, y_c, width, height = box
        pad_w, pad_h = pad
        x1 = (x_c - width / 2 - pad_w) / scale
        y1 = (y_c - height / 2 - pad_h) / scale
        x2 = (x_c + width / 2 - pad_w) / scale
        y2 = (y_c + height / 2 - pad_h) / scale
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w - 1, x2))
        y2 = max(0, min(image_h - 1, y2))
        return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))

    def _nms(self, boxes: List[Box], scores: List[float], iou_threshold: float) -> List[int]:
        if not boxes:
            return []
        boxes_array = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
        scores_array = np.array(scores, dtype=np.float32)
        indices = scores_array.argsort()[::-1]
        keep: List[int] = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            if indices.size == 1:
                break
            rest = indices[1:]
            ious = self._iou(boxes_array[i], boxes_array[rest])
            rest = rest[ious <= iou_threshold]
            indices = rest
        return keep

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union


def _resolve_cascade_params(mode_cfg: Dict[str, object], detection_cfg: Dict[str, object]) -> CascadeParams:
    cascade_override = detection_cfg.get("cascade_path") if detection_cfg else None
    if isinstance(cascade_override, str) and cascade_override:
        cascade_path = Path(cascade_override)
    else:
        override = mode_cfg.get("cascade")
        cascade_path = Path(str(override)) if override else default_cascade_path()
    return CascadeParams(
        cascade_path=cascade_path,
        scale_factor=float(mode_cfg.get("scale_factor", 1.1)),
        min_neighbors=int(mode_cfg.get("min_neighbors", 3)),
        min_size=int(mode_cfg.get("min_size", 60)),
    )


def _resolve_input_size(value: object) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
        return height, width
    size = int(value)
    return size, size


def create_detector(
    mode_cfg: Dict[str, object],
    detection_cfg: Optional[Dict[str, object]],
) -> BaseDetector:
    detection_cfg = detection_cfg or {}
    detector_type = str(detection_cfg.get("type", "cascade")).lower()
    if detector_type == "yolo":
        yolo_model = detection_cfg.get("yolo_model")
        if not yolo_model:
            raise ValueError("YOLO detector requires 'yolo_model' path in config.")
        model_path = Path(str(yolo_model)).expanduser()
        input_size = _resolve_input_size(detection_cfg.get("yolo_input_size", 320))
        class_ids = detection_cfg.get("yolo_class_ids")
        conf_threshold = detection_cfg.get("yolo_conf_threshold", 0.25)
        iou_threshold = detection_cfg.get("yolo_iou_threshold", 0.45)
        providers = detection_cfg.get("onnx_providers")
        if isinstance(providers, str):
            providers = [providers]
        return YoloOnnxDetector(
            model_path=model_path,
            input_size=input_size,
            class_ids=class_ids,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            providers=providers,
        )
    cascade_params = _resolve_cascade_params(mode_cfg, detection_cfg)
    return CascadeDetector(cascade_params)
