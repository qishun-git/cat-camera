from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from cat_face.embedding_model import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer
from cat_face.utils import ensure_dir, load_label_map, load_project_config, preprocess_face, resolve_paths


app = FastAPI(title="Pi-Cam")
templates = Jinja2Templates(directory="templates")


def load_recognizer(config: Dict[str, Any], paths: Dict[str, Path]) -> tuple[Optional[EmbeddingRecognizer], Dict[int, str], int]:
    training_cfg = config.get("training", {})
    recog_cfg = config.get("recognition", {})
    models_dir = paths["models"]
    vision_cfg = {"face_size": 100} | config.get("vision", {})
    embedding_name = recog_cfg.get("embedding_model_filename", training_cfg.get("embedding_model_filename", "embeddings.npz"))
    labels_name = recog_cfg.get("labels_filename", training_cfg.get("labels_filename", "labels.json"))
    embedding_path = models_dir / embedding_name
    labels_path = models_dir / labels_name
    try:
        labels = load_label_map(labels_path)
        model = EmbeddingModel.load(embedding_path)
    except FileNotFoundError:
        return None, {}, int(vision_cfg["face_size"])
    extractor = EmbeddingExtractor(input_size=int(training_cfg.get("embedding_input_size", 224)))
    recognizer = EmbeddingRecognizer(
        model=model,
        extractor=extractor,
        threshold=float(recog_cfg.get("embedding_threshold", 0.75)),
    )
    return recognizer, labels, int(vision_cfg["face_size"])


class ManagerState:
    def __init__(self) -> None:
        self.config = load_project_config()
        self.paths = resolve_paths(self.config)
        base = self.paths["base"]
        recorder_cfg = self.config.get("recorder", {})
        processing_cfg = self.config.get("clip_processing", {})
        self.raw_clips_dir = Path(recorder_cfg.get("output_dir") or (base / "clips"))
        self.compressed_dir = Path(processing_cfg.get("clips_dir") or (base / "compressed_clips"))
        self.recognized_root = ensure_dir(base / "recognized_clips")
        self.unknown_root = ensure_dir(base / "unknown_clips")
        self.unlabeled_root = ensure_dir(self.paths["unlabeled"])
        self.training_root = ensure_dir(self.paths["training"])
        self.reject_root = ensure_dir(self.paths["reject"])
        self.recognizer, self.label_map, self.face_size = load_recognizer(self.config, self.paths)

    def refresh(self) -> None:
        """Reload config and recognizer if files change."""
        self.__init__()


STATE = ManagerState()




def safe_join(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status_code=404, detail="Invalid path")
    return candidate


def unique_move(src: Path, dest_dir: Path) -> Path:
    dest_dir = ensure_dir(dest_dir)
    dest = dest_dir / src.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.rename(dest)
    return dest


def sidecar_path_for_clip(clip_path: Path) -> Path:
    return clip_path.with_suffix(f"{clip_path.suffix}.json")


def load_sidecar_data(clip_path: Path) -> Optional[Dict[str, Any]]:
    sidecar = sidecar_path_for_clip(clip_path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def clip_entries(root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries
    for clip in sorted(root.glob("**/*.mp4")):
        rel_path = clip.relative_to(root)
        label = rel_path.parts[0] if len(rel_path.parts) > 1 else clip.parent.name
        sidecar = sidecar_path_for_clip(clip)
        summary_data = load_sidecar_data(clip)
        summary = None
        if summary_data:
            summary = {
                "label": summary_data.get("highlight_label"),
                "detections": summary_data.get("detections_total", 0),
                "recognized": summary_data.get("recognized_total", 0),
            }
        entries.append(
            {
                "label": label,
                "path": str(rel_path),
                "size_mb": clip.stat().st_size / (1024 * 1024),
                "has_sidecar": sidecar.exists(),
                "summary": summary,
            }
        )
    return entries


def list_unlabeled_folders(root: Path) -> List[Dict[str, Any]]:
    folders: List[Dict[str, Any]] = []
    if not root.exists():
        return folders
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        count = sum(1 for _ in folder.glob("*"))
        folders.append({"name": folder.name, "path": str(folder.relative_to(root)), "count": count})
    return folders


def known_labels() -> List[str]:
    labels = []
    for path in sorted(STATE.training_root.iterdir()):
        if path.is_dir():
            labels.append(path.name)
    return labels


def predict_label(image_path: Path) -> Optional[str]:
    if STATE.recognizer is None:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    processed = preprocess_face(image, size=(STATE.face_size, STATE.face_size))
    label_id, score = STATE.recognizer.predict(processed)
    if label_id == -1:
        return None
    name = STATE.label_map.get(label_id)
    if not name:
        return None
    return f"{name} ({score:.2f})"


def unlabeled_images(folder: Path) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for image in sorted(p for p in folder.iterdir() if p.is_file()):
        suggestion = predict_label(image)
        images.append(
            {
                "name": image.name,
                "relative": str(image.relative_to(folder)),
                "size_kb": image.stat().st_size / 1024,
                "suggestion": suggestion,
            }
        )
    return images


def cleanup_folder(folder: Path) -> None:
    try:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass


@app.get("/", response_class=HTMLResponse)
@app.get("/stream", response_class=HTMLResponse)
def stream_page(request: Request):
    streaming_cfg = STATE.config.get("streaming") or {}
    stream_url = streaming_cfg.get("public_url")
    return templates.TemplateResponse("stream.html", {"request": request, "stream_url": stream_url})


@app.get("/clips", response_class=HTMLResponse)
def view_clips(request: Request):
    recognized = clip_entries(STATE.recognized_root)
    unknown = clip_entries(STATE.unknown_root)
    return templates.TemplateResponse(
        "clips.html",
        {
            "request": request,
            "recognized": recognized,
            "unknown": unknown,
        },
    )


def resolve_clip(category: str, clip_rel: str) -> tuple[Path, Path]:
    if category == "recognized":
        root = STATE.recognized_root
    elif category == "unknown":
        root = STATE.unknown_root
    else:
        raise HTTPException(status_code=404, detail="Invalid clip category")
    clip_path = safe_join(root, clip_rel)
    if not clip_path.exists() or clip_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Clip not found")
    return root, clip_path


@app.get("/clips/{category}/{clip_rel:path}/video")
def stream_clip(category: str, clip_rel: str):
    _, clip_path = resolve_clip(category, clip_rel)
    return FileResponse(clip_path)


@app.get("/clips/{category}/{clip_rel:path}", response_class=HTMLResponse)
def clip_detail(request: Request, category: str, clip_rel: str):
    _, clip_path = resolve_clip(category, clip_rel)
    summary = load_sidecar_data(clip_path)
    video_url = request.url_for("stream_clip", category=category, clip_rel=clip_rel)
    return templates.TemplateResponse(
        "clip_detail.html",
        {
            "request": request,
            "clip": clip_path,
            "category": category,
            "clip_rel": clip_rel,
            "summary": summary,
            "labels": known_labels(),
            "video_url": video_url,
        },
    )


@app.post("/clips/{category}/{clip_rel:path}/move")
def move_clip(category: str, clip_rel: str, action: str = Form(...), label: str = Form("")):
    root, clip_path = resolve_clip(category, clip_rel)
    sidecar = sidecar_path_for_clip(clip_path)
    if action == "to_label":
        target_label = label.strip()
        if not target_label:
            raise HTTPException(status_code=400, detail="Label required")
        dest_dir = STATE.recognized_root / target_label
    elif action == "to_unknown":
        dest_dir = STATE.unknown_root
    else:
        raise HTTPException(status_code=400, detail="Unknown action")

    source_parent = clip_path.parent
    new_path = unique_move(clip_path, dest_dir)
    if sidecar.exists():
        dest_json = sidecar_path_for_clip(new_path)
        counter = 1
        while dest_json.exists():
            dest_json = dest_json.with_name(f"{new_path.stem}_{counter}{new_path.suffix}.json")
            counter += 1
        sidecar.rename(dest_json)
    cleanup_folder(source_parent)
    return RedirectResponse(url="/clips", status_code=303)


@app.get("/unlabeled", response_class=HTMLResponse)
def unlabeled_list(request: Request):
    folders = list_unlabeled_folders(STATE.unlabeled_root)
    return templates.TemplateResponse(
        "unlabeled.html",
        {
            "request": request,
            "folders": folders,
        },
    )


def resolve_unlabeled_folder(folder_rel: str) -> Path:
    folder_path = safe_join(STATE.unlabeled_root, folder_rel)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder_path


@app.get("/unlabeled/{folder_rel:path}/image/{filename}")
def serve_unlabeled_image(folder_rel: str, filename: str):
    folder_path = resolve_unlabeled_folder(folder_rel)
    image_path = safe_join(folder_path, filename)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/unlabeled/{folder_rel:path}", response_class=HTMLResponse)
def unlabeled_folder(request: Request, folder_rel: str):
    folder_path = resolve_unlabeled_folder(folder_rel)
    images = unlabeled_images(folder_path)
    return templates.TemplateResponse(
        "unlabeled_folder.html",
        {
            "request": request,
            "folder": folder_path,
            "folder_rel": folder_rel,
            "images": images,
            "labels": known_labels(),
        },
    )


def move_image_to_label(image_path: Path, label: str) -> None:
    dest_dir = STATE.training_root / label
    unique_move(image_path, dest_dir)


def reject_image(image_path: Path) -> None:
    unique_move(image_path, STATE.reject_root)


@app.post("/unlabeled/{folder_rel:path}/label-folder")
def label_folder(folder_rel: str, label: str = Form(...)):
    folder_path = resolve_unlabeled_folder(folder_rel)
    label = label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label required")
    for image in sorted(p for p in folder_path.iterdir() if p.is_file()):
        move_image_to_label(image, label)
    cleanup_folder(folder_path)
    return RedirectResponse(url="/unlabeled", status_code=303)


@app.post("/unlabeled/{folder_rel:path}/image/{filename}/action")
def image_action(folder_rel: str, filename: str, action: str = Form(...), label: str = Form("")):
    folder_path = resolve_unlabeled_folder(folder_rel)
    image_path = safe_join(folder_path, filename)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if action == "to_label":
        chosen_label = label.strip()
        if not chosen_label:
            raise HTTPException(status_code=400, detail="Label required")
        move_image_to_label(image_path, chosen_label)
    elif action == "reject":
        reject_image(image_path)
    else:
        raise HTTPException(status_code=400, detail="Unknown action")
    cleanup_folder(folder_path)
    return RedirectResponse(url=f"/unlabeled/{folder_rel}", status_code=303)
