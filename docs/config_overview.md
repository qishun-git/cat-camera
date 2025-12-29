# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Shared directories for collection, sorting, training, and recognition outputs. |
| `vision` | `camera_index`, `face_size`, `prefer_picamera2`, `picamera_resolution`, `picamera_fps` | Camera and preprocessing settings shared by the recorder, recognition, and clip processing. |
| `motion` | `history`, `var_threshold`, `dilation_iterations`, `trigger_ratio`, `min_area`, `warmup_frames` | Background subtraction parameters that decide when `record_cat_video.py` starts/stops recording. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions`, `per_label_limit` | UI knobs for `sort_unlabeled.py`; `per_label_limit` randomly caps each label’s images after sorting. |
| `training` | `labels_filename`, `embedding_model_filename`, `embedding_input_size` | Output filenames written by `train_embeddings.py` and the embedding backbone’s input size. |
| `recognition` | `embedding_threshold`, `embedding_model_filename`, `labels_filename` | Runtime thresholds and filenames consumed by `recognize_live.py`. |
| `detection` | `model`, `input_size`, `class_ids`, `conf_threshold`, `iou_threshold`, `providers` | YOLO (ONNX Runtime) detector configuration shared by recognition and clip processing. |
| `recorder` | `output_dir`, `min_duration`, `max_duration`, `cooldown`, `absence_grace`, `fps`, `codec` | Default settings for `record_cat_video.py` (clip destination, duration bounds, cooldown, absence grace period, encoder options). |
| `clip_processing` | `clips_dir`, `save_limit`, `training_refresh_count`, `recognition_margin`, `compression_crf`, `watch_interval`, `detection_interval`, `trim_padding_seconds` | Controls how `process_clips.py` and the clip watcher behave (clip source directory, optional per-clip sampling limit, how many frames to promote to training, recognition margin for auto-tagging, H.265 CRF, watcher poll interval, YOLO cadence when scanning clips, and how much padding to keep when trimming clips down to the cat’s appearance). |

Update `vision.face_size` once to keep recording, training, and recognition aligned. Point `detection.model` at the YOLO ONNX file you want to run (for example, a 640×640 `yolov5n` export) and adjust `input_size`, confidence, or providers to match your hardware. Train embeddings via `python train_embeddings.py`, then set `recognition.embedding_threshold` to a value that balances precision/recall for your cats. All clip collection now flows through `record_cat_video.py`, which triggers on motion first and relies on clip processing to confirm cats, so there’s no separate still-image capture step.

**Detection parameter reference**
- `model`: Filesystem path to the YOLO ONNX file. Relative paths resolve from the repo root (e.g., `models/yolov5n.onnx`).
- `input_size`: The resolution the ONNX model expects. Provide a single integer (`640`) or `[width, height]`; the detector letterboxes each frame to this size before inference.
- `class_ids`: Optional list of class indices to keep. COCO cats use id `15`, so `[15]` ignores all other detections. Leave empty (`[]`) to accept every YOLO class.
- `conf_threshold`: Minimum objectness × class score required to keep a detection. Increase to reduce false positives; decrease to detect harder examples.
- `iou_threshold`: Intersection-over-union threshold used during non-max suppression. Lower values keep more overlapping boxes; higher values prune more aggressively.
- `providers`: ONNX Runtime execution providers (e.g., `["CPUExecutionProvider"]`, `["CoreMLExecutionProvider"]`). Leave empty to let onnxruntime choose defaults for your platform.

**Motion parameter reference**
- `history`: Number of frames the background model keeps. Increase if motion is consistently slow; decrease if the scene changes often.
- `var_threshold`: Higher values make the detector less sensitive to small pixel changes.
- `dilation_iterations`: Morphological dilations applied to the foreground mask to close gaps before contour detection.
- `trigger_ratio`: Minimum fraction of pixels that must change before we even inspect contours.
- `min_area`: Minimum contour area (in pixels) to count as motion. Tune to ignore small flickers/noise.
- `warmup_frames`: Number of initial frames ignored while the background model stabilizes after startup.
- `blur_kernel`: Optional Gaussian blur kernel (odd) applied before background subtraction. Set to 0/1 to disable; small kernels (3–5) help suppress high-frequency vibration noise.

**Clip processing notes**
- `detection_interval`: YOLO cadence (seconds) while scanning stored clips; raising it reduces CPU load during batch processing.
- `trim_padding_seconds`: Amount of footage to keep before the first cat detection and after the last; `process_clips.py` trims each clip down to cat activity plus this padding.

**Recorder parameter reference**
- `output_dir`: Where `record_cat_video.py` stores completed clips. Relative paths resolve from the repo root (e.g., `data/clips`).
- `min_duration`: Minimum clip length in seconds; recordings shorter than this are discarded.
- `max_duration`: Maximum clip length in seconds; recording stops automatically once this is reached.
- `cooldown`: Time to wait (seconds) after saving a clip before another recording may begin.
- `absence_grace`: Grace period (seconds) after the last detection before a clip is closed. If the cat leaves before `min_duration`, the clip is discarded.
- `fps`: Preferred output FPS for saved clips. Set to `0` (or omit) to reuse the camera’s FPS.
- `codec`: FourCC string passed to OpenCV’s `VideoWriter` (for MP4 output, use `mp4v`, `avc1`, etc.).
