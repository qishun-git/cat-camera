# `configs/recognize.yaml`

| Key | Description | Typical values |
| --- | --- | --- |
| `model_path` | Path to a previously trained LBPH model XML. | `"models/lbph_model.xml"` |
| `labels_path` | Path to the JSON label map produced during training. | `"models/labels.json"` |
| `camera_index` | VideoCapture index/device path for live inference. | `0` |
| `cascade` | Optional Haar/LBP cascade XML path; blank defaults to OpenCV’s cat-face cascade. | `""` |
| `size` | Image resize value that must match `capture`/`train` configs. | `100` |
| `threshold` | Maximum acceptable LBPH confidence (predictions above this become `unknown`). | `80.0` |
| `scale_factor` | Cascade detection scale factor. | `1.1` |
| `min_neighbors` | Cascade minNeighbors parameter. | `3` |
| `min_size` | Minimum face width/height in pixels. | `60` |
