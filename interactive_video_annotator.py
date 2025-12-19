#!/usr/bin/env python3.11
"""
Interactive Video Annotation with Frame-by-Frame Correction
Annotate one frame, then review and correct each propagated frame
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add SAM2 to path
sys.path.append('./segment-anything-2')
from sam2.build_sam import build_sam2_video_predictor

class InteractiveVideoAnnotator:
    def __init__(self, video_path, start_frame, checkpoint_path, model_cfg):
        self.video_path = video_path
        self.start_frame = start_frame
        self.frame_dir = "frames"
        self.current_frame = start_frame
        
        # Extract frames
        print(f"Extracting frames from {video_path}...")
        self.extract_frames()
        
        # Load SAM2 video predictor
        print("Loading SAM2 video predictor...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        
        # Initialize inference state
        print("Initializing video state...")
        self.inference_state = self.predictor.init_state(video_path=str(Path(self.frame_dir).absolute()))
        
        # Points storage for initial frame
        self.frame_points = {}  # {frame_idx: {'positive': [], 'negative': []}}
        self.frame_points[start_frame] = {'positive': [], 'negative': []}
        
        # Video segments storage
        self.video_segments = {}
        
        # State
        self.mode = 'annotate'  # 'annotate' or 'propagate'
        self.propagate_direction = 'forward'  # 'forward' or 'backward'
        self.auto_advance = False  # Auto-advance to next frame if True
        
        print(f"\nStarting on frame {start_frame}")
        self.show_instructions()
    
    def show_instructions(self):
        print("=" * 70)
        print("ANNOTATION MODE:")
        print("  LEFT CLICK   - Add positive point (GREEN) - what you WANT")
        print("  RIGHT CLICK  - Add negative point (RED) - what you DON'T want")
        print("  'r' - Reset points on current frame")
        print("  'c' - Clear last point")
        print("  SPACE - Accept and go to NEXT frame")
        print("  'a' - Toggle auto-advance (skip confirmation)")
        print("  'b' - Go BACKWARD one frame")
        print("  'd' - Switch propagation direction (forward/backward)")
        print("  's' - Save and quit")
        print("  'q' - Quit without saving")
        print("=" * 70)
    
    def extract_frames(self):
        """Extract frames from video"""
        output_path = Path(self.frame_dir)
        output_path.mkdir(exist_ok=True)
        
        # Check if frames already exist
        existing_frames = list(output_path.glob("*.jpg"))
        if len(existing_frames) > 0:
            print(f"Using existing {len(existing_frames)} frames in {self.frame_dir}/")
            cap = cv2.VideoCapture(self.video_path)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return
        
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
    
    def load_frame(self, frame_idx):
        """Load a frame image"""
        frame_path = Path(self.frame_dir) / f"{frame_idx:05d}.jpg"
        return cv2.imread(str(frame_path))
    
    def get_current_points(self):
        """Get positive and negative points for current frame"""
        if self.current_frame not in self.frame_points:
            self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        return self.frame_points[self.current_frame]
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add positive point
            points = self.get_current_points()
            points['positive'].append([x, y])
            self.update_segmentation()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add negative point
            points = self.get_current_points()
            points['negative'].append([x, y])
            self.update_segmentation()
    
    def update_segmentation(self):
        """Update segmentation for current frame"""
        points = self.get_current_points()
        
        if len(points['positive']) == 0:
            self.display_frame()
            return
        
        # Combine positive and negative points
        all_points = []
        labels = []
        
        for pt in points['positive']:
            all_points.append(pt)
            labels.append(1)
        
        for pt in points['negative']:
            all_points.append(pt)
            labels.append(0)
        
        all_points = np.array(all_points)
        labels = np.array(labels)
        
        # Add points to SAM2
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame,
            obj_id=1,
            points=all_points,
            labels=labels,
        )
        
        # Store the mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        self.video_segments[self.current_frame] = {1: mask}
        
        # Display
        self.display_frame()
    
    def display_frame(self):
        """Display current frame with mask and points"""
        frame = self.load_frame(self.current_frame)
        display = frame.copy()
        
        # Draw mask if available
        if self.current_frame in self.video_segments:
            masks = self.video_segments[self.current_frame]
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                mask_bool = mask_2d.astype(bool)
                display[mask_bool] = display[mask_bool] * 0.5 + np.array([255, 100, 0]) * 0.5
        
        # Draw points
        points = self.get_current_points()
        
        for pt in points['positive']:
            cv2.circle(display, tuple(pt), 7, (0, 255, 0), -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 2)
        
        for pt in points['negative']:
            cv2.circle(display, tuple(pt), 7, (0, 0, 255), -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 2)
        
        # Add info overlay
        info_bg = display.copy()
        cv2.rectangle(info_bg, (0, 0), (display.shape[1], 120), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, info_bg, 0.3, 0)
        
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames-1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(display, f"Pos: {len(points['positive'])} | Neg: {len(points['negative'])}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        direction_text = f"Direction: {self.propagate_direction.upper()}"
        if self.auto_advance:
            direction_text += " | AUTO"
        cv2.putText(display, direction_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.current_frame in self.video_segments:
            mask_area = np.sum(self.video_segments[self.current_frame][1])
            cv2.putText(display, f"Area: {mask_area:.0f} px", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Interactive Video Annotator', display)
    
    def propagate_one_frame(self):
        """Propagate to the next frame and show result"""
        # Determine next frame
        if self.propagate_direction == 'forward':
            next_frame = self.current_frame + 1
            if next_frame >= self.total_frames:
                print("Reached end of video (forward)")
                return False
        else:  # backward
            next_frame = self.current_frame - 1
            if next_frame < 0:
                print("Reached start of video (backward)")
                return False
        
        # Propagate in video (SAM2 handles this internally)
        print(f"Propagating to frame {next_frame}...")
        
        # SAM2's propagate_in_video returns all frames, but we only need the next one
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=next_frame, max_frame_num_to_track=1
        ):
            if out_frame_idx == next_frame:
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                self.video_segments[next_frame] = {1: mask}
                break
        
        # Move to next frame
        self.current_frame = next_frame
        
        # Initialize points for this frame if not exists
        if self.current_frame not in self.frame_points:
            self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        
        # Display
        self.display_frame()
        
        return True
    
    def go_backward_one(self):
        """Go back one frame"""
        prev_frame = self.current_frame - 1
        if prev_frame < 0:
            print("Already at first frame")
            return
        
        self.current_frame = prev_frame
        
        # If we haven't segmented this frame yet, propagate to it
        if self.current_frame not in self.video_segments:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=self.current_frame, max_frame_num_to_track=1
            ):
                if out_frame_idx == self.current_frame:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    self.video_segments[self.current_frame] = {1: mask}
                    break
        
        self.display_frame()
    
    def reset_current_frame(self):
        """Reset points on current frame"""
        self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        self.display_frame()
        print(f"Reset points on frame {self.current_frame}")
    
    def clear_last_point(self):
        """Remove last added point"""
        points = self.get_current_points()
        
        if len(points['negative']) > 0:
            points['negative'].pop()
            print("Removed last negative point")
        elif len(points['positive']) > 0:
            points['positive'].pop()
            print("Removed last positive point")
        
        self.update_segmentation()
    
    def save_results(self):
        """Save all annotated frames"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        print("\nSaving results...")
        
        # Save individual masks
        for frame_idx in sorted(self.video_segments.keys()):
            masks = self.video_segments[frame_idx]
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                mask_img = (mask_2d * 255).astype(np.uint8)
                cv2.imwrite(str(masks_dir / f"mask_{frame_idx:05d}.png"), mask_img)
        
        # Create output video
        if len(self.video_segments) > 0:
            first_frame_idx = min(self.video_segments.keys())
            first_frame = self.load_frame(first_frame_idx)
            h, w = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter('annotated_video.mp4', fourcc, self.fps, (w, h))
            
            for frame_idx in sorted(self.video_segments.keys()):
                frame = self.load_frame(frame_idx)
                masks = self.video_segments[frame_idx]
                
                overlay = frame.copy()
                for obj_id, mask in masks.items():
                    mask_2d = mask.squeeze()
                    mask_bool = mask_2d.astype(bool)
                    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 100, 0]) * 0.5
                
                result = overlay.astype(np.uint8)
                mask_area = np.sum(mask_2d)
                cv2.putText(result, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Area: {mask_area:.0f} px", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out_video.write(result)
            
            out_video.release()
            print(f"✓ Saved {len(self.video_segments)} annotated frames to annotated_video.mp4")
        
        print(f"✓ Saved masks to {masks_dir}/")
    
    def run(self):
        """Run the interactive annotation loop"""
        cv2.namedWindow('Interactive Video Annotator')
        cv2.setMouseCallback('Interactive Video Annotator', self.mouse_callback)
        
        self.display_frame()
        
        while True:
            key = cv2.waitKey(1 if self.auto_advance else 0) & 0xFF
            
            if key == ord('q'):
                print("Quitting without saving...")
                break
                
            elif key == ord('s'):
                print("Saving and quitting...")
                self.save_results()
                break
                
            elif key == ord(' '):  # Space - advance to next frame
                if self.get_current_points()['positive'] or self.current_frame in self.video_segments:
                    success = self.propagate_one_frame()
                    if not success and self.propagate_direction == 'forward':
                        # Switch to backward
                        self.propagate_direction = 'backward'
                        self.current_frame = self.start_frame
                        print("Switched to backward propagation")
                        self.display_frame()
                else:
                    print("Add at least one positive point first!")
                    
            elif key == ord('b'):  # Go backward one frame
                self.go_backward_one()
                
            elif key == ord('r'):  # Reset
                self.reset_current_frame()
                
            elif key == ord('c'):  # Clear last
                self.clear_last_point()
                
            elif key == ord('d'):  # Switch direction
                self.propagate_direction = 'backward' if self.propagate_direction == 'forward' else 'forward'
                print(f"Switched to {self.propagate_direction} propagation")
                self.display_frame()
                
            elif key == ord('a'):  # Toggle auto-advance
                self.auto_advance = not self.auto_advance
                print(f"Auto-advance: {'ON' if self.auto_advance else 'OFF'}")
                self.display_frame()
            
            # Auto-advance logic
            if self.auto_advance and self.current_frame in self.video_segments:
                cv2.waitKey(100)  # Brief pause to see the frame
                self.propagate_one_frame()
        
        cv2.destroyAllWindows()

def main():
    video_path = "input_2.mp4"
    start_frame = 5
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    try:
        annotator = InteractiveVideoAnnotator(video_path, start_frame, checkpoint_path, model_cfg)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
