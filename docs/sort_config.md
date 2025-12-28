# `configs/sort.yaml`

| Key | Description | Typical values |
| --- | --- | --- |
| `unlabeled_root` | Directory populated by unlabeled capture sessions (e.g., `data/unlabeled`). | `"data/unlabeled"` |
| `destination_root` | Root folder containing labeled class directories (same as the training data root, e.g., `data/training`). The sorter always scans this to build numeric shortcuts automatically. | `"data/training"` |
| `delete_rejects` | When `true`, pressing the delete shortcut removes a frame permanently; when `false`, it gets moved into `reject_folder`. | `false` |
| `reject_folder` | Subdirectory (under `destination_root`) that stores discarded frames when not deleting. | `"rejected"` |
| `window_name` | Title for the OpenCV preview window. | `"Sort Unlabeled"` |
| `image_extensions` | File extensions to include when scanning unlabeled directories. | `[".png", ".jpg"]` |
| `window_width` / `window_height` | Desired size for the preview window; OpenCV scales images accordingly. | `640`, `480` |
| `window_x` / `window_y` | Screen coordinates where the preview window should appear. | `100`, `100` |
