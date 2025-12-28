# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Shared directories for capture, sorting, training, and recognition outputs. |
| `vision` | `camera_index`, `face_size` | Shared capture/recognition camera index and face crop size (in pixels). |
| `capture` | `mode`, `display_window`, `auto_session_subfolders`, `max_images_per_cat`, `max_unlabeled_images` | Controls how `capture_faces.py` saves data and rotates session storage. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions` | UI knobs for `sort_unlabeled.py`; uses the shared `paths` automatically. |
| `training` | `labels_filename`, `embedding_model_filename`, `embedding_input_size` | Output filenames written by `train_embeddings.py` and the embedding backbone’s input size. |
| `recognition` | `embedding_threshold`, `embedding_model_filename`, `labels_filename` | Runtime thresholds and filenames consumed by `recognize_live.py`. |
| `detection` | `model`, `input_size`, `class_ids`, `conf_threshold`, `iou_threshold`, `providers` | YOLO (ONNX Runtime) detector configuration shared by capture and recognition. |

Update `vision.face_size` once to keep capture, training, and recognition aligned. Point `detection.model` at the YOLO ONNX file you want to run (for example, a 640×640 `yolov5n` export) and adjust `input_size`, confidence, or providers to match your hardware. Train embeddings via `python train_embeddings.py`, then set `recognition.embedding_threshold` to a value that balances precision/recall for your cats. When running `capture_faces.py` in labeled mode, supply the cat name via `--cat-name` (for example, `python capture_faces.py --cat-name whiskers`); unlabeled sessions omit that argument.
