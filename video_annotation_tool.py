#!/usr/bin/env python3.11
"""
Video Annotation Tool with SAM2
1. Annotate one frame with positive/negative points
2. Propagate to all frames in the video
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add SAM2 to path
sys.path.append('./segment-anything-2')
from sam2.build_sam import build_sam2_video_predictor

class VideoAnnotationTool:
    def __init__(self, video_path, start_frame, checkpoint_path, model_cfg):
        self.video_path = video_path
        self.start_frame = start_frame
        self.frame_dir = "frames"
        
        # Extract frames
        print(f"Extracting frames from {video_path}...")
        self.extract_frames()
        
        # Load the annotation frame
        frame_path = Path(self.frame_dir) / f"{start_frame:05d}.jpg"
        self.image = cv2.imread(str(frame_path))
        if self.image is None:
            raise ValueError(f"Could not load frame: {frame_path}")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()
        
        # Points storage
        self.positive_points = []
        self.negative_points = []
        self.mask = None
        
        # SAM2 will be loaded later when propagating
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.predictor = None
        
        print(f"\nAnnotating frame {start_frame}")
        print("=" * 60)
        print("Controls:")
        print("  LEFT CLICK  - Add positive point (green) - what you WANT")
        print("  RIGHT CLICK - Add negative point (red) - what you DON'T want")
        print("  'r' - Reset all points")
        print("  'c' - Clear last point")
        print("  'p' - PROPAGATE to all frames (forward & backward)")
        print("  'q' - Quit")
        print("=" * 60)
    
    def extract_frames(self):
        """Extract frames from video"""
        output_path = Path(self.frame_dir)
        output_path.mkdir(exist_ok=True)
        
        # Clear existing frames
        for file in output_path.glob("*.jpg"):
            file.unlink()
        
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.total_frames} frames @ {self.fps:.2f} fps")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = output_path / f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {frame_idx} frames to {self.frame_dir}/")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add positive point
            self.positive_points.append([x, y])
            self.redraw()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add negative point
            self.negative_points.append([x, y])
            self.redraw()
    
    def redraw(self):
        """Redraw the display with points"""
        self.display_image = self.image.copy()
        
        # Draw positive points (green)
        for pt in self.positive_points:
            cv2.circle(self.display_image, tuple(pt), 7, (0, 255, 0), -1)
            cv2.circle(self.display_image, tuple(pt), 8, (255, 255, 255), 2)
        
        # Draw negative points (red)
        for pt in self.negative_points:
            cv2.circle(self.display_image, tuple(pt), 7, (0, 0, 255), -1)
            cv2.circle(self.display_image, tuple(pt), 8, (255, 255, 255), 2)
        
        # Add info
        info_text = f"Frame {self.start_frame} | Positive: {len(self.positive_points)} | Negative: {len(self.negative_points)}"
        cv2.putText(self.display_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Press 'p' to propagate to all frames", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Video Annotation', self.display_image)
    
    def reset(self):
        """Reset all points"""
        self.positive_points = []
        self.negative_points = []
        self.redraw()
        print("Reset all points")
    
    def clear_last(self):
        """Remove the last added point"""
        if len(self.negative_points) > 0:
            self.negative_points.pop()
            print("Removed last negative point")
        elif len(self.positive_points) > 0:
            self.positive_points.pop()
            print("Removed last positive point")
        self.redraw()
    
    def propagate_to_video(self):
        """Propagate annotation to all frames using SAM2 video predictor"""
        if len(self.positive_points) == 0:
            print("Error: Add at least one positive point first!")
            return
        
        print("\n" + "=" * 60)
        print("PROPAGATING TO ALL FRAMES")
        print("=" * 60)
        
        # Load SAM2 video predictor
        print("Loading SAM2 video predictor...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=device)
        
        # Initialize inference state
        print("Initializing video state...")
        inference_state = self.predictor.init_state(video_path=str(Path(self.frame_dir).absolute()))
        
        # Prepare points
        points = []
        labels = []
        
        for pt in self.positive_points:
            points.append(pt)
            labels.append(1)  # Positive
        
        for pt in self.negative_points:
            points.append(pt)
            labels.append(0)  # Negative
        
        points = np.array(points)
        labels = np.array(labels)
        
        # Add prompts on the start frame
        print(f"Adding {len(points)} points on frame {self.start_frame}...")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=self.start_frame,
            obj_id=1,
            points=points,
            labels=labels,
        )
        
        # Propagate through video
        print("Propagating through all frames...")
        print("This will process both forward and backward from frame", self.start_frame)
        
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            if out_frame_idx % 10 == 0:
                print(f"  Processed frame {out_frame_idx}/{self.total_frames}")
        
        print(f"✓ Propagated to {len(video_segments)} frames")
        
        # Save results
        self.save_results(video_segments)
        
        # Show preview
        self.show_results_preview(video_segments)
    
    def save_results(self, video_segments):
        """Save all masks and create output video"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        print("\nSaving results...")
        
        # Create output video
        first_frame = cv2.imread(f"{self.frame_dir}/00000.jpg")
        h, w = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('annotated_video.mp4', fourcc, self.fps, (w, h))
        
        for frame_idx in sorted(video_segments.keys()):
            # Load frame
            frame_path = Path(self.frame_dir) / f"{frame_idx:05d}.jpg"
            frame = cv2.imread(str(frame_path))
            
            # Get mask
            masks = video_segments[frame_idx]
            
            # Create overlay
            overlay = frame.copy()
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                mask_bool = mask_2d.astype(bool)
                
                # Blue overlay
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 100, 0]) * 0.5
                
                # Save individual mask
                mask_img = (mask_2d * 255).astype(np.uint8)
                cv2.imwrite(str(masks_dir / f"mask_{frame_idx:05d}.png"), mask_img)
            
            result = overlay.astype(np.uint8)
            
            # Add frame info
            mask_area = np.sum(mask_2d)
            cv2.putText(result, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, f"Area: {mask_area:.0f} px", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out_video.write(result)
        
        out_video.release()
        
        print(f"✓ Saved annotated video: annotated_video.mp4")
        print(f"✓ Saved individual masks: {masks_dir}/")
        
        # Compute and save metrics
        self.save_metrics(video_segments)
    
    def save_metrics(self, video_segments):
        """Compute and save area metrics"""
        import matplotlib.pyplot as plt
        
        frame_indices = sorted(video_segments.keys())
        areas = []
        
        for frame_idx in frame_indices:
            masks = video_segments[frame_idx]
            total_area = sum(np.sum(mask) for mask in masks.values())
            areas.append(total_area)
        
        times = np.array(frame_indices) / self.fps
        areas = np.array(areas)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(times, areas, 'b-', linewidth=2)
        plt.axvline(self.start_frame / self.fps, color='r', linestyle='--', label=f'Annotation Frame {self.start_frame}')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Segmented Area (pixels)', fontsize=12)
        plt.title('Segmentation Area Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rate of change
        if len(areas) > 1:
            area_rate = np.gradient(areas, times)
            
            plt.subplot(1, 2, 2)
            plt.plot(times, area_rate, 'r-', linewidth=2)
            plt.axvline(self.start_frame / self.fps, color='r', linestyle='--', label=f'Annotation Frame {self.start_frame}')
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Rate of Change (pixels/sec)', fontsize=12)
            plt.title('Segmentation Rate', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/segmentation_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved analysis plot: output/segmentation_analysis.png")
    
    def show_results_preview(self, video_segments):
        """Show a scrollable preview of results"""
        print("\n" + "=" * 60)
        print("PREVIEW MODE - Use arrow keys to navigate frames")
        print("Left/Right arrows: Previous/Next frame")
        print("'q': Quit preview")
        print("=" * 60)
        
        frame_indices = sorted(video_segments.keys())
        current_idx = 0
        
        while True:
            frame_num = frame_indices[current_idx]
            
            # Load frame
            frame_path = Path(self.frame_dir) / f"{frame_num:05d}.jpg"
            frame = cv2.imread(str(frame_path))
            
            # Get mask
            masks = video_segments[frame_num]
            
            # Create overlay
            overlay = frame.copy()
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                mask_bool = mask_2d.astype(bool)
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 100, 0]) * 0.5
            
            result = overlay.astype(np.uint8)
            
            # Add info
            mask_area = np.sum(mask_2d)
            cv2.putText(result, f"Frame: {frame_num}/{self.total_frames-1} ({current_idx+1}/{len(frame_indices)})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, f"Area: {mask_area:.0f} px", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, "Arrow keys: Navigate | 'q': Quit", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Results Preview', result)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # Left arrow
                current_idx = max(0, current_idx - 1)
            elif key == 83 or key == 3:  # Right arrow
                current_idx = min(len(frame_indices) - 1, current_idx + 1)
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Run the interactive annotation loop"""
        cv2.namedWindow('Video Annotation')
        cv2.setMouseCallback('Video Annotation', self.mouse_callback)
        
        self.redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('c'):
                self.clear_last()
            elif key == ord('p'):
                cv2.destroyAllWindows()
                self.propagate_to_video()
                break
        
        cv2.destroyAllWindows()

def main():
    # Configuration
    video_path = "input_4.mp4"
    start_frame = 5
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    try:
        tool = VideoAnnotationTool(video_path, start_frame, checkpoint_path, model_cfg)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
