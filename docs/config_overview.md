# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Shared directories for capture, sorting, training, and recognition outputs. |
| `vision` | `camera_index`, `face_size` | Shared capture/recognition camera index and face crop size (in pixels). |
| `capture` | `mode`, `display_window`, `auto_session_subfolders`, `max_images_per_cat`, `max_unlabeled_images` | Controls how `capture_faces.py` saves data and rotates session storage. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions`, `per_label_limit` | UI knobs for `sort_unlabeled.py`; `per_label_limit` randomly caps each label’s images after sorting. |
| `training` | `labels_filename`, `embedding_model_filename`, `embedding_input_size` | Output filenames written by `train_embeddings.py` and the embedding backbone’s input size. |
| `recognition` | `embedding_threshold`, `embedding_model_filename`, `labels_filename` | Runtime thresholds and filenames consumed by `recognize_live.py`. |
| `detection` | `model`, `input_size`, `class_ids`, `conf_threshold`, `iou_threshold`, `providers` | YOLO (ONNX Runtime) detector configuration shared by capture, recognition, and clip recording. |
| `recorder` | `output_dir`, `min_duration`, `max_duration`, `cooldown`, `absence_grace`, `fps`, `codec`, `show_window` | Default settings for `record_cat_video.py` (clip destination, duration bounds, cooldown, absence grace period, preview window). |
| `clip_processing` | `clips_dir`, `save_limit` | Controls how `process_clips.py` processes saved videos (input directory, optional per-clip limit; clips always save into the unlabeled pool). |

Update `vision.face_size` once to keep capture, training, and recognition aligned. Point `detection.model` at the YOLO ONNX file you want to run (for example, a 640×640 `yolov5n` export) and adjust `input_size`, confidence, or providers to match your hardware. Train embeddings via `python train_embeddings.py`, then set `recognition.embedding_threshold` to a value that balances precision/recall for your cats. When running `capture_faces.py` in labeled mode, supply the cat name via `--cat-name` (for example, `python capture_faces.py --cat-name whiskers`); unlabeled sessions omit that argument.

**Detection parameter reference**
- `model`: Filesystem path to the YOLO ONNX file. Relative paths resolve from the repo root (e.g., `models/yolov5n.onnx`).
- `input_size`: The resolution the ONNX model expects. Provide a single integer (`640`) or `[width, height]`; the detector letterboxes each frame to this size before inference.
- `class_ids`: Optional list of class indices to keep. COCO cats use id `15`, so `[15]` ignores all other detections. Leave empty (`[]`) to accept every YOLO class.
- `conf_threshold`: Minimum objectness × class score required to keep a detection. Increase to reduce false positives; decrease to detect harder examples.
- `iou_threshold`: Intersection-over-union threshold used during non-max suppression. Lower values keep more overlapping boxes; higher values prune more aggressively.
- `providers`: ONNX Runtime execution providers (e.g., `["CPUExecutionProvider"]`, `["CoreMLExecutionProvider"]`). Leave empty to let onnxruntime choose defaults for your platform.

**Recorder parameter reference**
- `output_dir`: Where `record_cat_video.py` stores completed clips. Relative paths resolve from the repo root (e.g., `data/clips`).
- `min_duration`: Minimum clip length in seconds; recordings shorter than this are discarded.
- `max_duration`: Maximum clip length in seconds; recording stops automatically once this is reached.
- `cooldown`: Time to wait (seconds) after saving a clip before another recording may begin.
- `absence_grace`: Grace period (seconds) after the last detection before a clip is closed. If the cat leaves before `min_duration`, the clip is discarded.
- `fps`: Preferred output FPS for saved clips. Set to `0` (or omit) to reuse the camera’s FPS.
- `codec`: FourCC string passed to OpenCV’s `VideoWriter` (for MP4 output, use `mp4v`, `avc1`, etc.).
- `show_window`: Boolean that toggles the live preview window during recording.
