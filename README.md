# Video Annotation Tool with SAM2

Interactive video annotation using Meta's Segment Anything 2 (SAM2). Uses union mode where masks can only grow - ideal for tracking fluid flow.

## Setup

```bash
# Create virtual environment (requires Python 3.11+ with tkinter)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SAM2
cd segment-anything-2 && pip install -e . && cd ..
```

## Download Model Checkpoints

Download SAM2 checkpoints to `checkpoints/`:
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) (recommended)
- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) (faster)

## Usage

```bash
source venv/bin/activate
python annotate.py --video input_4.mp4
```

### Options
- `--video, -v`: Video file path (default: input_4.mp4)
- `--model, -m`: Model size - tiny, small, large (default: large)

### Controls
| Key | Action |
|-----|--------|
| Left click | Add point (include region) |
| Right click | Exclude region |
| Space | Next frame (propagate mask) |
| B | Go back one frame |
| A | Auto-propagate to end |
| N | Mark frame as "no fluid" |
| C | Clear last point |
| R | Reset current frame |
| S | Save and quit |
| Esc/X | Stop auto-propagation |
| Q | Quit without saving |

## Output

- `output/masks/` - Individual mask images
- `annotated_video_union.mp4` - Video with mask overlay
