#!/usr/bin/env python3.11
"""
Simple Interactive Segmentation with SAM2
Click to add positive (green) and negative (red) points
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add SAM2 to path
sys.path.append('./segment-anything-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class InteractiveSegmenter:
    def __init__(self, image_path, checkpoint_path, model_cfg):
        # Load image
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()
        
        # Initialize SAM2
        print("Loading SAM2 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        # Set the image
        print("Processing image...")
        self.predictor.set_image(self.image_rgb)
        
        # Points storage
        self.positive_points = []  # Points of interest
        self.negative_points = []  # Points to exclude
        self.current_mode = 'positive'  # 'positive' or 'negative'
        self.mask = None
        
        print("\nReady! Click on the image to segment.")
        print("=" * 60)
        print("Controls:")
        print("  LEFT CLICK  - Add positive point (green) - what you WANT")
        print("  RIGHT CLICK - Add negative point (red) - what you DON'T want")
        print("  'r' - Reset all points")
        print("  'c' - Clear last point")
        print("  's' - Save mask")
        print("  'q' - Quit")
        print("=" * 60)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add positive point
            self.positive_points.append([x, y])
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            self.update_segmentation()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add negative point
            self.negative_points.append([x, y])
            cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
            self.update_segmentation()
    
    def update_segmentation(self):
        """Run SAM2 segmentation with current points"""
        if len(self.positive_points) == 0:
            return
        
        # Combine points and labels
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
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # Use the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        self.mask = masks[best_mask_idx]
        
        # Update display
        self.display_mask()
    
    def display_mask(self):
        """Display the current mask overlay"""
        # Start with original image
        self.display_image = self.image.copy()
        
        if self.mask is not None:
            # Create blue overlay for masked region
            overlay = self.display_image.copy()
            mask_bool = self.mask.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 100, 0]) * 0.5
            self.display_image = overlay.astype(np.uint8)
        
        # Draw positive points (green)
        for pt in self.positive_points:
            cv2.circle(self.display_image, tuple(pt), 7, (0, 255, 0), -1)
            cv2.circle(self.display_image, tuple(pt), 8, (255, 255, 255), 2)
        
        # Draw negative points (red)
        for pt in self.negative_points:
            cv2.circle(self.display_image, tuple(pt), 7, (0, 0, 255), -1)
            cv2.circle(self.display_image, tuple(pt), 8, (255, 255, 255), 2)
        
        # Add info text
        info_text = f"Positive: {len(self.positive_points)} | Negative: {len(self.negative_points)}"
        cv2.putText(self.display_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.mask is not None:
            area = np.sum(self.mask)
            cv2.putText(self.display_image, f"Area: {area:.0f} px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Interactive Segmentation', self.display_image)
    
    def reset(self):
        """Reset all points and mask"""
        self.positive_points = []
        self.negative_points = []
        self.mask = None
        self.display_image = self.image.copy()
        cv2.imshow('Interactive Segmentation', self.display_image)
        print("Reset all points")
    
    def clear_last(self):
        """Remove the last added point"""
        if len(self.negative_points) > 0:
            self.negative_points.pop()
            print("Removed last negative point")
        elif len(self.positive_points) > 0:
            self.positive_points.pop()
            print("Removed last positive point")
        
        # Re-segment if we still have points
        self.display_image = self.image.copy()
        if len(self.positive_points) > 0:
            self.update_segmentation()
        else:
            self.display_mask()
    
    def save_mask(self):
        """Save the current mask"""
        if self.mask is None:
            print("No mask to save!")
            return
        
        # Save mask as binary image
        mask_img = (self.mask * 255).astype(np.uint8)
        mask_path = "segmentation_mask.png"
        cv2.imwrite(mask_path, mask_img)
        
        # Save overlay
        overlay_path = "segmentation_overlay.png"
        cv2.imwrite(overlay_path, self.display_image)
        
        # Save masked region only
        masked_region = self.image.copy()
        masked_region[~self.mask] = 0
        masked_path = "segmentation_masked.png"
        cv2.imwrite(masked_path, masked_region)
        
        print(f"\n✓ Saved:")
        print(f"  - Binary mask: {mask_path}")
        print(f"  - Overlay: {overlay_path}")
        print(f"  - Masked region: {masked_path}")
    
    def run(self):
        """Run the interactive loop"""
        cv2.namedWindow('Interactive Segmentation')
        cv2.setMouseCallback('Interactive Segmentation', self.mouse_callback)
        
        # Show initial image
        cv2.imshow('Interactive Segmentation', self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('c'):
                self.clear_last()
            elif key == ord('s'):
                self.save_mask()
        
        cv2.destroyAllWindows()

def main():
    # Configuration
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Check for image
    if Path("input_2.mp4").exists():
        # Extract first frame from video
        print("Extracting frame from video...")
        cap = cv2.VideoCapture("input_2.mp4")
        
        # Get frame 5 (as requested earlier)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            image_path = "frame_5.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Using frame 5 from video: {image_path}")
        else:
            print("Error: Could not extract frame from video")
            return
    elif Path("frame_5.jpg").exists():
        image_path = "frame_5.jpg"
    else:
        print("Error: No input image found!")
        print("Please provide 'input_2.mp4' or 'frame_5.jpg'")
        return
    
    # Run interactive segmentation
    try:
        segmenter = InteractiveSegmenter(image_path, checkpoint_path, model_cfg)
        segmenter.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
