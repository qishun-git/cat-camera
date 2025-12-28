# Unified configuration (`configs/cat_face.yaml`)

| Section | Keys | Description |
| --- | --- | --- |
| `paths` | `base_data_dir`, `unlabeled_dir`, `training_dir`, `reject_dir`, `models_dir` | Defines the directories shared across capture, sorting, training, and recognition. |
| `capture` | `mode`, `camera_index`, `cascade`, `size`, `scale_factor`, `min_neighbors`, `min_size`, `display_window`, `auto_session_subfolders`, `max_images_per_cat`, `max_unlabeled_images` | Controls how `capture_faces.py` captures and rotates images. |
| `sorter` | `window_*`, `delete_rejects`, `image_extensions` | Configures the sorter's UI and delete behavior; it automatically uses `paths.unlabeled_dir` and `paths.training_dir` for source/destination. |
| `training` | `size`, `radius`, `neighbors`, `grid_x`, `grid_y`, `threshold`, `model_filename`, `labels_filename` | LBPH hyperparameters and filenames stored under `paths.models_dir`. |
| `recognition` | `camera_index`, `cascade`, `size`, `threshold`, `scale_factor`, `min_neighbors`, `min_size` | Recognition settings used by `recognize_live.py`; again, paths are derived from the shared section. |

When running `capture_faces.py` in labeled mode you now provide the subject's name via the `--cat-name` CLI option (for example `python capture_faces.py --cat-name whiskers`). Unlabeled sessions do not require this argument.
