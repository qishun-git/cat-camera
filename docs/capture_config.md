# `configs/capture.yaml`

| Key | Description | Typical values |
| --- | --- | --- |
| `cat_name` | Folder/label name used when saving new samples. | `"whiskers"` |
| `output` | Root directory containing class subfolders. A folder named after `cat_name` is created inside. | `"data"` |
| `camera_index` | Integer index (0,1,…) or device path passed to `cv2.VideoCapture`. | `0` |
| `cascade` | Path to a Haar/LBP cascade XML file. Leave blank to auto-load OpenCV’s `haarcascade_frontalcatface.xml`. | `""` |
| `size` | Square size (pixels) to which each detected face is resized before saving. | `100` |
| `scale_factor` | Cascade detector scale factor; lower values improve detection at the cost of speed. | `1.1` |
| `min_neighbors` | Cascade minNeighbors parameter (higher reduces false positives). | `3` |
| `min_size` | Minimum face width/height in pixels to consider a detection valid. | `60` |
| `display_window` | If `true`, shows the OpenCV preview and requires pressing SPACE to save a frame. If `false`, saves automatically whenever a face is detected. | `true` |
