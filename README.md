# SAM2 Flow Tracker

**Video segmentation and object tracking with automatic COCO/YOLO annotation export**

A powerful toolkit for tracking objects (water flow, materials, etc.) through video using Meta's Segment Anything Model 2 (SAM2). Features interactive annotation, automatic propagation, and exports in industry-standard formats.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![SAM2](https://img.shields.io/badge/SAM-2.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Interactive Annotation** - Click to mark positive/negative points on any frame
- **Video Propagation** - Automatically track objects across all video frames
- **COCO Export** - Full annotation JSON with polygons, bboxes, and areas
- **YOLO Export** - Per-frame label files ready for training
- **Flow Metrics** - Area tracking, seepage rate analysis, and visualization
- **Real-time Preview** - Watch segmentation as it propagates

---

## Project Structure

```
SAM2_Flow_Tracker/
├── flow_tracker.py                 # Main video tracking pipeline
├── interactive_segmentation.py     # Single-image annotation tool
├── interactive_video_annotator.py  # Frame-by-frame video annotation
├── video_annotation_tool.py        # Batch video annotation
├── requirements.txt                # Python dependencies
├── checkpoints/                    # SAM2 model weights
│   └── sam2.1_hiera_large.pt
├── segment-anything-2/             # SAM2 library
├── tools/
│   └── create_demo_video.py        # Generate demo videos
├── examples/
│   └── demo_seepage.mp4            # Example video
├── frames/                         # Extracted video frames (auto-generated)
└── output/                         # Results (auto-generated)
    ├── annotations_coco.json       # COCO format annotations
    ├── labels_yolo/                # YOLO format labels
    │   ├── 00000.txt
    │   ├── 00001.txt
    │   └── classes.txt
    ├── flow_analysis.png           # Metrics visualization
    └── frame_*.jpg                 # Sample output frames
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SAM2_Flow_Tracker.git
cd SAM2_Flow_Tracker

# Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# Install SAM2
cd segment-anything-2
pip install -e .
cd ..
```

### 2. Download SAM2 Checkpoint

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SAM2.1 Hiera-Large (856 MB)
curl -L -o checkpoints/sam2.1_hiera_large.pt \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
```

### 3. Run the Tracker

```bash
# Using default demo video
python3.11 flow_tracker.py

# Or with your own video (place as input_2.mp4)
python3.11 flow_tracker.py
```

---

## Tools

### Flow Tracker (Main Pipeline)
```bash
python3.11 flow_tracker.py
```
**Workflow:**
1. Extracts frames from video
2. Opens frame 5 for interactive point selection
3. Click to mark the object to track
4. Press Enter to confirm
5. Propagates segmentation through all frames
6. Exports COCO + YOLO annotations
7. Generates metrics and visualization

### Interactive Segmentation (Single Image)
```bash
python3.11 interactive_segmentation.py
```
**Controls:**
- **Left Click** - Add positive point (include this region)
- **Right Click** - Add negative point (exclude this region)
- **R** - Reset all points
- **C** - Clear last point
- **S** - Save mask
- **Q** - Quit

### Video Annotator (Frame-by-Frame)
```bash
python3.11 interactive_video_annotator.py
```
**Controls:**
- **Left/Right Click** - Add positive/negative points
- **Space** - Propagate to next frame
- **B** - Go back one frame
- **S** - Save all annotations
- **Q** - Quit

---

## Output Formats

### COCO Format (`output/annotations_coco.json`)

```json
{
  "info": {
    "description": "SAM2 Water Flow Tracking",
    "date_created": "2024-12-19T18:00:00"
  },
  "images": [
    {"id": 0, "file_name": "00000.jpg", "height": 720, "width": 1280}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [100, 200, 150, 80],
      "area": 12000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "water", "supercategory": "liquid"}
  ]
}
```

### YOLO Format (`output/labels_yolo/`)

Each frame gets a `.txt` file with normalized bounding boxes:

```
# 00000.txt
# class_id  x_center  y_center  width  height
0 0.512340 0.345678 0.123456 0.089012
```

A `classes.txt` file is also generated:
```
water
```

---

## Metrics Output

The flow tracker generates:
- **Area vs Time Plot** - Shows object growth over time
- **Seepage Rate Plot** - Rate of change (pixels/sec)
- **Statistics:**
  - Initial/final area
  - Total growth percentage
  - Average and max rates
  - Duration

---

## Input Formats

Supported video formats:
- MP4 (recommended)
- AVI
- MOV
- Any format supported by OpenCV

Place your video as `input_2.mp4` in the project root, or modify `flow_tracker.py` to specify a custom path.

---

## Configuration

### Change Model Size

Edit the model config in the Python files:

```python
# Available models (largest to smallest):
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"      # Large (default)
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"     # Base+
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"      # Small
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"      # Tiny
```

### Change Starting Frame

```python
prompt_frame_idx = 5  # Frame to annotate (0-indexed)
```

### Custom Categories

Edit the export functions in `flow_tracker.py` to add more classes:

```python
"categories": [
    {"id": 1, "name": "water", "supercategory": "liquid"},
    {"id": 2, "name": "oil", "supercategory": "liquid"},
]
```

---

## Troubleshooting

### Python Version
```bash
# Check version (requires 3.11+)
python3.11 --version
```

### Video Won't Load
```bash
# Check file format
file input_2.mp4
# Should show: "ISO Media, MP4" not "HTML document"
```

### Memory Issues
- Use a smaller model (`sam2.1_hiera_t.yaml`)
- Process shorter video clips
- Reduce frame resolution

### CUDA / GPU
- SAM2 auto-detects CUDA
- On Apple Silicon, MPS is used automatically
- CPU processing works but is slower

---

## References

- [SAM2 Paper](https://arxiv.org/abs/2408.00714) - Segment Anything in Images and Videos
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2) - Official repository
- [COCO Format](https://cocodataset.org/#format-data) - Annotation standard
- [YOLO Format](https://docs.ultralytics.com/datasets/detect/) - Label format

---

## License

This project is for research and educational purposes. SAM2 model weights are subject to Meta's license terms.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Built with SAM2 by Meta AI**
