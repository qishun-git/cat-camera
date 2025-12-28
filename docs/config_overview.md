# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Defines the directories shared across capture, sorting, training, and recognition. |
| `capture` | `mode`, `camera_index`, `cascade`, `size`, `scale_factor`, `min_neighbors`, `min_size`, `display_window`, `auto_session_subfolders`, `max_images_per_cat`, `max_unlabeled_images` | Controls how `capture_faces.py` captures and rotates images. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions` | Configures the sorter's UI and delete behavior; it automatically uses `paths.unlabeled_dir` and `paths.training_dir` for source/destination. |
| `training` | `radius`, `neighbors`, `grid_x`, `grid_y`, `threshold`, `model_filename`, `labels_filename`, `validation_split` | LBPH hyperparameters, filenames under `paths.models_dir`, and the optional validation split used for accuracy reporting. |
| `recognition` | `camera_index`, `cascade`, `size`, `threshold`, `scale_factor`, `min_neighbors`, `min_size` | Recognition settings used by `recognize_live.py`; again, paths are derived from the shared section. |
| `detection` | `type`, `cascade_path`, `yolo_model`, `yolo_input_size`, `yolo_class_ids`, `yolo_conf_threshold`, `yolo_iou_threshold`, `onnx_providers` | Chooses the detector backend (OpenCV cascade vs. ONNX YOLO) and shared parameters for both capture and recognition. |

Capture, training, and recognition all reuse the `capture.size` value, so update it in one place to keep preprocessing consistent. Switch between cascade and YOLO detection by updating the `detection` section (for example, set `type: yolo` and point `yolo_model` at an exported YOLOv5n ONNX file). When running `capture_faces.py` in labeled mode you now provide the subject's name via the `--cat-name` CLI option (for example `python capture_faces.py --cat-name whiskers`). Unlabeled sessions do not require this argument.
