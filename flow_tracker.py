#!/usr/bin/env python3.11
"""
Water Flow Tracking using SAM2
Tracks water seepage through materials using Segment Anything Model 2
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Add SAM2 to path
sys.path.append('./segment-anything-2')

from sam2.build_sam import build_sam2_video_predictor

def setup_sam2(checkpoint_path, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """Initialize SAM2 video predictor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build predictor - config path should be relative to SAM2 package
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    return predictor

def extract_frames(video_path, output_dir="frames"):
    """Extract frames from video"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clear existing frames
    for file in output_path.glob("*.jpg"):
        file.unlink()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_count} frames @ {fps:.2f} fps")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = output_path / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}/")
    return frame_idx, fps

def get_user_prompt(frame_path, frame_number=0):
    """Display specified frame and get user to select point for tracking"""
    img = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"\n=== Interactive Prompt Selection (Frame {frame_number}) ===")
    print("Click on the water/wet region you want to track")
    print("Press 'q' to quit, 'r' to reset, 'Enter' to confirm")
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img_rgb, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Select Point', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    cv2.namedWindow('Select Point')
    cv2.setMouseCallback('Select Point', mouse_callback)
    cv2.imshow('Select Point', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):
            points = []
            img_rgb = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
            cv2.imshow('Select Point', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        elif key == 13:  # Enter
            break
    
    cv2.destroyAllWindows()
    
    if len(points) == 0:
        print("No points selected. Using center of frame as default.")
        h, w = img.shape[:2]
        points = [[w//2, h//2]]
    
    return np.array(points)

def track_flow(predictor, frame_dir, prompt_points, prompt_frame_idx=0, output_dir="output", show_preview=True):
    """Track water flow using SAM2"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=str(Path(frame_dir).absolute()))
    
    # Add prompts for specified frame
    frame_idx = prompt_frame_idx
    object_id = 1
    
    # Positive points (where water is)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=object_id,
        points=prompt_points,
        labels=np.ones(len(prompt_points), dtype=np.int32),  # 1 = positive
    )
    
    print(f"Added {len(prompt_points)} prompt points at frame {frame_idx}")
    
    # Propagate through video
    print("Propagating masks through video...")
    if show_preview:
        print("Press 'q' to quit preview, 'p' to pause/resume")
    
    video_segments = {}
    paused = False
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        if out_frame_idx % 10 == 0:
            print(f"  Processed frame {out_frame_idx}")
        
        # Show live preview
        if show_preview:
            # Load the frame
            frame_path = Path(frame_dir) / f"{out_frame_idx:05d}.jpg"
            frame = cv2.imread(str(frame_path))
            
            if frame is not None:
                # Get mask for this frame
                masks = video_segments[out_frame_idx]
                
                # Create colored overlay
                overlay = frame.copy()
                for obj_id, mask in masks.items():
                    mask_2d = mask.squeeze()
                    # Blue overlay for water
                    overlay[mask_2d] = overlay[mask_2d] * 0.5 + np.array([255, 100, 0]) * 0.5
                
                result = overlay.astype(np.uint8)
                
                # Add frame info
                mask_area = np.sum(mask_2d)
                cv2.putText(result, f"Frame: {out_frame_idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Area: {mask_area:.0f} px", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, "Press 'q' to quit, 'p' to pause", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display
                cv2.imshow('Flow Tracking Preview', result)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nPreview closed by user")
                    cv2.destroyAllWindows()
                    show_preview = False
                elif key == ord('p'):
                    paused = not paused
                    print(f"\n{'Paused' if paused else 'Resumed'}")
                
                # If paused, wait for unpause
                while paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        paused = False
                        print("Resumed")
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        show_preview = False
                        paused = False
    
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"Tracked {len(video_segments)} frames")
    return video_segments

def visualize_results(frame_dir, video_segments, output_dir="output", output_video="flow_tracking.mp4"):
    """Create visualization with masks overlaid"""
    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    
    if len(frame_files) == 0:
        print("No frames found!")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
    
    # Process each frame
    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        
        if frame_idx in video_segments:
            # Get mask for this frame
            masks = video_segments[frame_idx]
            
            # Create colored overlay
            overlay = frame.copy()
            for obj_id, mask in masks.items():
                # Squeeze mask to 2D
                mask_2d = mask.squeeze()
                
                # Create blue overlay for water
                overlay[mask_2d] = overlay[mask_2d] * 0.5 + np.array([255, 100, 0]) * 0.5
            
            frame = overlay.astype(np.uint8)
            
            # Add frame number and stats
            mask_area = np.sum(mask_2d)
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {mask_area:.0f} px", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        
        # Save sample frames
        if frame_idx % 30 == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_idx:05d}.jpg", frame)
    
    out.release()
    print(f"\nOutput video saved: {output_video}")

def mask_to_coco_polygon(mask):
    """Convert binary mask to COCO polygon format"""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.size >= 6:  # Need at least 3 points
            contour = contour.flatten().tolist()
            polygons.append(contour)
    
    return polygons

def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x, y, width, height]"""
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return [0, 0, 0, 0]
    
    ymin, ymax = pos[0].min(), pos[0].max()
    xmin, xmax = pos[1].min(), pos[1].max()
    
    return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]

def export_coco_format(video_segments, frame_dir, output_path="output/annotations_coco.json"):
    """Export segmentation results in COCO format"""
    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    img_height, img_width = first_frame.shape[:2]
    
    coco_output = {
        "info": {
            "description": "SAM2 Water Flow Tracking",
            "date_created": datetime.now().isoformat(),
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "water", "supercategory": "liquid"}
        ]
    }
    
    annotation_id = 1
    
    for frame_idx, frame_file in enumerate(frame_files):
        # Add image info
        coco_output["images"].append({
            "id": frame_idx,
            "file_name": frame_file.name,
            "height": img_height,
            "width": img_width,
            "frame_index": frame_idx
        })
        
        # Add annotations if available
        if frame_idx in video_segments:
            masks = video_segments[frame_idx]
            
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze().astype(np.uint8)
                
                # Get polygon
                polygons = mask_to_coco_polygon(mask_2d)
                
                # Get bounding box
                bbox = mask_to_bbox(mask_2d)
                
                # Calculate area
                area = int(np.sum(mask_2d))
                
                if area > 0 and len(polygons) > 0:
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": frame_idx,
                        "category_id": 1,  # water
                        "segmentation": polygons,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # Save JSON
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"COCO annotations saved: {output_path}")
    print(f"  - {len(coco_output['images'])} images")
    print(f"  - {len(coco_output['annotations'])} annotations")
    
    return output_path

def export_yolo_format(video_segments, frame_dir, output_dir="output/labels_yolo"):
    """Export segmentation results in YOLO format (one txt file per frame)"""
    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clear existing labels
    for file in output_path.glob("*.txt"):
        file.unlink()
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    img_height, img_width = first_frame.shape[:2]
    
    annotation_count = 0
    
    for frame_idx, frame_file in enumerate(frame_files):
        # Create label file for this frame
        label_path = output_path / f"{frame_file.stem}.txt"
        
        with open(label_path, 'w') as f:
            if frame_idx in video_segments:
                masks = video_segments[frame_idx]
                
                for obj_id, mask in masks.items():
                    mask_2d = mask.squeeze().astype(np.uint8)
                    
                    # Get bounding box
                    bbox = mask_to_bbox(mask_2d)
                    x, y, w, h = bbox
                    
                    if w > 0 and h > 0:
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width_norm = w / img_width
                        height_norm = h / img_height
                        
                        # YOLO format: class_id x_center y_center width height
                        # class_id 0 = water
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                        annotation_count += 1
    
    print(f"YOLO labels saved: {output_dir}/")
    print(f"  - {len(frame_files)} label files")
    print(f"  - {annotation_count} total annotations")
    
    # Create classes.txt file
    classes_path = output_path / "classes.txt"
    with open(classes_path, 'w') as f:
        f.write("water\n")
    print(f"  - Classes file: {classes_path}")
    
    return output_dir


def analyze_flow_metrics(video_segments, fps):
    """Compute flow metrics over time"""
    frame_indices = sorted(video_segments.keys())
    areas = []
    
    for frame_idx in frame_indices:
        masks = video_segments[frame_idx]
        total_area = sum(np.sum(mask) for mask in masks.values())
        areas.append(total_area)
    
    # Plot area over time
    times = np.array(frame_indices) / fps
    areas = np.array(areas)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, areas, 'b-', linewidth=2)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Wet Area (pixels)', fontsize=12)
    plt.title('Water Seepage Area Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Compute rate of change
    if len(areas) > 1:
        area_rate = np.gradient(areas, times)
        
        plt.subplot(1, 2, 2)
        plt.plot(times, area_rate, 'r-', linewidth=2)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Seepage Rate (pixels/sec)', fontsize=12)
        plt.title('Water Seepage Rate', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/flow_analysis.png', dpi=150, bbox_inches='tight')
    print("Flow analysis plot saved: output/flow_analysis.png")
    
    # Print statistics
    print("\n=== Flow Metrics ===")
    print(f"Initial area: {areas[0]:.0f} pixels")
    print(f"Final area: {areas[-1]:.0f} pixels")
    print(f"Total growth: {areas[-1] - areas[0]:.0f} pixels ({((areas[-1]/areas[0] - 1)*100):.1f}%)")
    print(f"Duration: {times[-1]:.2f} seconds")
    if len(areas) > 1:
        print(f"Average rate: {np.mean(area_rate):.1f} pixels/sec")
        print(f"Max rate: {np.max(area_rate):.1f} pixels/sec")

def main():
    print("=" * 60)
    print("Water Flow Tracking with SAM2")
    print("=" * 60)
    
    # Paths
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Relative to SAM2 package
    frame_dir = "frames"
    output_dir = "output"
    
    # Check for video files
    if Path("input_2.mp4").exists() and Path("input_2.mp4").stat().st_size > 1000000:  # > 1MB
        video_path = "input_2.mp4"
        print(f"Using video: {video_path}")
    elif Path("demo_seepage.mp4").exists():
        video_path = "demo_seepage.mp4"
        print(f"Using demo video: {video_path}")
    else:
        print("Error: No valid video found. Run create_demo_video.py first.")
        return
    
    # Extract frames
    print("\n[1/5] Extracting frames from video...")
    num_frames, fps = extract_frames(video_path, frame_dir)
    
    # Get user prompt on frame 5
    prompt_frame_idx = 5
    print(f"\n[2/5] Getting tracking prompt (Frame {prompt_frame_idx})...")
    prompt_frame_file = f"{frame_dir}/{prompt_frame_idx:05d}.jpg"
    prompt_points = get_user_prompt(prompt_frame_file, prompt_frame_idx)
    
    if prompt_points is None:
        print("Cancelled by user.")
        return
    
    print(f"Using prompt points on frame {prompt_frame_idx}: {prompt_points}")
    
    # Setup SAM2
    print("\n[3/5] Loading SAM2 model...")
    predictor = setup_sam2(checkpoint_path, model_cfg)
    
    # Track flow
    print("\n[4/5] Tracking water flow...")
    video_segments = track_flow(predictor, frame_dir, prompt_points, prompt_frame_idx, output_dir)
    
    # Visualize
    print("\n[5/5] Creating visualization...")
    visualize_results(frame_dir, video_segments, output_dir, "flow_tracking.mp4")
    
    # Export annotations
    print("\n[6/7] Exporting COCO format annotations...")
    export_coco_format(video_segments, frame_dir, "output/annotations_coco.json")
    
    print("\n[7/7] Exporting YOLO format annotations...")
    export_yolo_format(video_segments, frame_dir, "output/labels_yolo")
    
    # Analyze
    print("\n[8/8] Analyzing flow metrics...")
    analyze_flow_metrics(video_segments, fps)
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print(f"Output video: flow_tracking.mp4")
    print(f"Analysis plot: {output_dir}/flow_analysis.png")
    print(f"Sample frames: {output_dir}/frame_*.jpg")
    print(f"COCO annotations: {output_dir}/annotations_coco.json")
    print(f"YOLO labels: {output_dir}/labels_yolo/*.txt")


if __name__ == "__main__":
    main()
