# `configs/train.yaml`

| Key | Description | Typical values |
| --- | --- | --- |
| `data_dir` | Root directory where each cat has its own subfolder of preprocessed images. | `"data"` |
| `model_path` | Destination file for the trained LBPH model (`.xml`). | `"models/lbph_model.xml"` |
| `labels_path` | JSON file storing the integer label to class-name mapping. | `"models/labels.json"` |
| `size` | Image size expected by the recognizer; must match the capture/preprocessing size. | `100` |
| `radius` | LBPH radius parameter (larger values capture more texture context). | `2` |
| `neighbors` | LBPH neighbors parameter; higher values smooth the histogram. | `8` |
| `grid_x` | Number of grid divisions along the X axis for LBPH histograms. | `8` |
| `grid_y` | Number of grid divisions along the Y axis for LBPH histograms. | `8` |
| `threshold` | Maximum confidence value accepted during recognition (lower is stricter). Stored with the model for reference. | `80.0` |
