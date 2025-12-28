# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Defines the directories shared across capture, sorting, training, and recognition. |
| `vision` | `camera_index`, `face_size` | Shared camera/preprocessing settings that feed capture, training, and recognition. |
| `capture` | `mode`, `display_window`, `auto_session_subfolders`, `max_images_per_cat`, `max_unlabeled_images` | Controls how `capture_faces.py` captures and rotates images. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions` | Configures the sorter's UI and delete behavior; it automatically uses `paths.unlabeled_dir` and `paths.training_dir` for source/destination. |
| `training` | `radius`, `neighbors`, `grid_x`, `grid_y`, `threshold`, `model_filename`, `labels_filename`, `validation_split`, `embedding_model_filename`, `embedding_input_size` | LBPH hyperparameters, filenames under `paths.models_dir`, optional validation split, and filenames/input sizes for the embedding recognizer trained via `train_embeddings.py`. |
| `recognition` | `method`, `threshold`, `embedding_threshold`, `embedding_model_filename` | Recognition settings used by `recognize_live.py`; set `method: embedding` to switch to the MobileNet+cosine pipeline. |
| `detection` | `type`, `cascade.*`, `yolo.*` | Chooses the detector backend (OpenCV cascade vs. ONNX YOLO) and per-backend parameters shared by capture and recognition. |

Capture, training, and recognition all reuse the `vision.face_size` value, so update it in one place to keep preprocessing consistent. Switch between cascade and YOLO detection by updating the `detection` section (for example, set `type: yolo` and point `detection.yolo.model` at an exported YOLOv5n ONNX file). To try the embedding recognizer: run `python train_embeddings.py` to produce `models/embeddings.npz`, set `recognition.method: embedding`, and adjust `recognition.embedding_threshold` as needed. When running `capture_faces.py` in labeled mode you now provide the subject's name via the `--cat-name` CLI option (for example `python capture_faces.py --cat-name whiskers`). Unlabeled sessions do not require this argument.
