#!/usr/bin/env python3.11
"""
Interactive Video Annotation with SAM2 - Tkinter UI
Start at frame 0, add points when fluid appears, propagate forward
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

# Add SAM2 to path
sys.path.append('./segment-anything-2')
from sam2.build_sam import build_sam2_video_predictor


class VideoAnnotatorApp:
    def __init__(self, root, video_path, checkpoint_path, model_cfg):
        self.root = root
        self.root.title("Video Annotation Tool - SAM2")
        self.root.configure(bg='#2b2b2b')

        self.video_path = video_path
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.frame_dir = "frames"
        self.current_frame = 0

        # State
        self.positive_points = []
        self.negative_points = []
        self.video_segments = {}
        self.frame_points = {}
        self.predictor = None
        self.inference_state = None
        self.total_frames = 0
        self.fps = 30
        self.display_scale = 1.0
        self.is_loading = False

        # Setup UI
        self._setup_ui()

        # Start initialization in background
        self.root.after(100, self._start_initialization)

    def _setup_ui(self):
        """Setup the Tkinter UI"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Green.TButton', foreground='green')
        style.configure('Red.TButton', foreground='red')

        # Top info bar
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=(0, 10))

        self.frame_label = ttk.Label(self.info_frame, text="Frame: 0/0", style='Title.TLabel')
        self.frame_label.pack(side=tk.LEFT)

        self.points_label = ttk.Label(self.info_frame, text="Points: +0 / -0", style='TLabel')
        self.points_label.pack(side=tk.LEFT, padx=20)

        self.area_label = ttk.Label(self.info_frame, text="", style='TLabel')
        self.area_label.pack(side=tk.RIGHT)

        # Canvas for video display
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self._on_left_click)
        self.canvas.bind('<Button-2>', self._on_right_click)  # Middle click on Mac
        self.canvas.bind('<Button-3>', self._on_right_click)  # Right click

        # Progress bar
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)

        self.progress_label = ttk.Label(self.progress_frame, text="Ready", style='Status.TLabel')
        self.progress_label.pack(side=tk.TOP, anchor=tk.W)

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=10)

        # Left side - Point controls
        self.point_controls = ttk.Frame(self.button_frame)
        self.point_controls.pack(side=tk.LEFT)

        ttk.Label(self.point_controls, text="Click: Left=Positive, Right=Negative",
                 style='Status.TLabel').pack(side=tk.LEFT)

        ttk.Button(self.point_controls, text="Clear Last (C)",
                  command=self._clear_last_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.point_controls, text="Reset All (R)",
                  command=self._reset_points).pack(side=tk.LEFT, padx=5)

        # Right side - Navigation
        self.nav_controls = ttk.Frame(self.button_frame)
        self.nav_controls.pack(side=tk.RIGHT)

        ttk.Button(self.nav_controls, text="Back (B)",
                  command=self._go_back).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_controls, text="Next Frame (Space)",
                  command=self._next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_controls, text="Save & Quit (S)",
                  command=self._save_and_quit).pack(side=tk.LEFT, padx=5)

        # Status bar at bottom
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(self.status_frame, text="Initializing...", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT)

        # Keyboard bindings
        self.root.bind('<space>', lambda e: self._next_frame())
        self.root.bind('r', lambda e: self._reset_points())
        self.root.bind('c', lambda e: self._clear_last_point())
        self.root.bind('s', lambda e: self._save_and_quit())
        self.root.bind('q', lambda e: self._quit())
        self.root.bind('<Escape>', lambda e: self._quit())
        self.root.bind('b', lambda e: self._go_back())

    def _start_initialization(self):
        """Start initialization in a background thread"""
        self.is_loading = True
        thread = threading.Thread(target=self._initialize, daemon=True)
        thread.start()

    def _initialize(self):
        """Initialize SAM2 and extract frames"""
        try:
            # Step 1: Extract frames
            self._update_progress("Extracting video frames...", 0)
            self._extract_frames()

            # Step 2: Load SAM2
            self._update_progress("Loading SAM2 model (this may take a moment)...", 33)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=device)

            # Step 3: Initialize video state
            self._update_progress("Initializing video state...", 66)
            self.inference_state = self.predictor.init_state(
                video_path=str(Path(self.frame_dir).absolute())
            )

            # Done
            self._update_progress("Ready! Click to add points, then press Space for next frame.", 100)
            self.is_loading = False

            # Show first frame
            self.root.after(0, self._display_frame)

        except Exception as e:
            self._update_progress(f"Error: {e}", 0)
            self.is_loading = False

    def _update_progress(self, message, percent):
        """Update progress bar and label (thread-safe)"""
        def update():
            self.progress_label.config(text=message)
            self.progress_bar['value'] = percent
            self.status_label.config(text=message)
        self.root.after(0, update)

    def _set_status(self, message):
        """Set status message"""
        self.status_label.config(text=message)

    def _extract_frames(self):
        """Extract frames from video"""
        output_path = Path(self.frame_dir)
        output_path.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if frames already exist
        existing_frames = list(output_path.glob("*.jpg"))
        if len(existing_frames) >= self.total_frames:
            cap.release()
            return

        # Extract frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = output_path / f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1

            if frame_idx % 20 == 0:
                progress = int((frame_idx / self.total_frames) * 30)
                self._update_progress(f"Extracting frames: {frame_idx}/{self.total_frames}", progress)

        cap.release()

    def _load_frame(self, frame_idx):
        """Load a frame as PIL Image"""
        frame_path = Path(self.frame_dir) / f"{frame_idx:05d}.jpg"
        if not frame_path.exists():
            return None
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _display_frame(self):
        """Display current frame with mask and points"""
        if self.is_loading:
            return

        frame = self._load_frame(self.current_frame)
        if frame is None:
            return

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw mask if available
        if self.current_frame in self.video_segments:
            masks = self.video_segments[self.current_frame]
            for obj_id, mask in masks.items():
                mask_2d = mask.squeeze()
                mask_bool = mask_2d.astype(bool)
                # Blue overlay
                overlay_color = np.array([100, 150, 255])
                display[mask_bool] = (display[mask_bool] * 0.5 + overlay_color * 0.5).astype(np.uint8)

        # Draw points
        points = self._get_current_points()
        for pt in points['positive']:
            cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
            cv2.circle(display, tuple(pt), 9, (255, 255, 255), 2)
        for pt in points['negative']:
            cv2.circle(display, tuple(pt), 8, (255, 0, 0), -1)
            cv2.circle(display, tuple(pt), 9, (255, 255, 255), 2)

        # Convert to PhotoImage
        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            scale_w = canvas_w / w
            scale_h = canvas_h / h
            self.display_scale = min(scale_w, scale_h, 1.0)

            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            display = cv2.resize(display, (new_w, new_h))

        img = Image.fromarray(display)
        self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            anchor=tk.CENTER,
            image=self.photo
        )

        # Update labels
        self.frame_label.config(text=f"Frame: {self.current_frame}/{self.total_frames - 1}")
        self.points_label.config(text=f"Points: +{len(points['positive'])} / -{len(points['negative'])}")

        if self.current_frame in self.video_segments:
            mask_area = np.sum(self.video_segments[self.current_frame][1])
            self.area_label.config(text=f"Area: {mask_area:,.0f} px")
        else:
            self.area_label.config(text="")

    def _get_current_points(self):
        """Get points for current frame"""
        if self.current_frame not in self.frame_points:
            self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        return self.frame_points[self.current_frame]

    def _canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates"""
        # Get canvas center offset
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        frame = self._load_frame(self.current_frame)
        if frame is None:
            return None, None

        h, w = frame.shape[:2]
        disp_w = int(w * self.display_scale)
        disp_h = int(h * self.display_scale)

        # Calculate offset (image is centered)
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2

        # Convert to image coordinates
        img_x = int((x - offset_x) / self.display_scale)
        img_y = int((y - offset_y) / self.display_scale)

        # Check bounds
        if 0 <= img_x < w and 0 <= img_y < h:
            return img_x, img_y
        return None, None

    def _on_left_click(self, event):
        """Handle left click - add positive point"""
        if self.is_loading:
            return

        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        if img_x is None:
            return

        points = self._get_current_points()
        points['positive'].append([img_x, img_y])
        self._set_status(f"Added POSITIVE point at ({img_x}, {img_y})")
        self._update_segmentation()

    def _on_right_click(self, event):
        """Handle right click - add negative point"""
        if self.is_loading:
            return

        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        if img_x is None:
            return

        points = self._get_current_points()
        points['negative'].append([img_x, img_y])
        self._set_status(f"Added NEGATIVE point at ({img_x}, {img_y})")
        self._update_segmentation()

    def _update_segmentation(self):
        """Run SAM2 segmentation with current points"""
        points = self._get_current_points()

        if len(points['positive']) == 0:
            self._display_frame()
            return

        # Combine points
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

        # Run SAM2
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame,
            obj_id=1,
            points=all_points,
            labels=labels,
        )

        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        self.video_segments[self.current_frame] = {1: mask}

        self._display_frame()

    def _next_frame(self):
        """Propagate to next frame"""
        if self.is_loading:
            return

        points = self._get_current_points()
        if not points['positive'] and self.current_frame not in self.video_segments:
            self._set_status("Add at least one positive point first!")
            return

        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self._set_status("Reached end of video! Press S to save.")
            return

        # Run propagation in background
        self.is_loading = True
        self._update_progress(f"Propagating to frame {next_frame}...", 50)

        def propagate():
            try:
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=next_frame, max_frame_num_to_track=1
                ):
                    if out_frame_idx == next_frame:
                        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                        self.video_segments[next_frame] = {1: mask}
                        break

                self.current_frame = next_frame
                if self.current_frame not in self.frame_points:
                    self.frame_points[self.current_frame] = {'positive': [], 'negative': []}

                self.is_loading = False
                self.root.after(0, lambda: self._update_progress(
                    f"Frame {next_frame} - Click to refine or press Space for next", 100))
                self.root.after(0, self._display_frame)

            except Exception as e:
                self.is_loading = False
                self.root.after(0, lambda: self._set_status(f"Error: {e}"))

        thread = threading.Thread(target=propagate, daemon=True)
        thread.start()

    def _go_back(self):
        """Go back to previous frame (only if already segmented)"""
        if self.is_loading:
            return

        if self.current_frame <= 0:
            self._set_status("Already at first frame")
            return

        prev_frame = self.current_frame - 1

        # Only go back to frames we've already processed
        if prev_frame in self.video_segments:
            self.current_frame = prev_frame
            self._set_status(f"Went back to frame {prev_frame}")
            self._display_frame()
        else:
            self._set_status(f"Frame {prev_frame} not yet processed - can only go back to processed frames")

    def _clear_last_point(self):
        """Remove last added point"""
        if self.is_loading:
            return

        points = self._get_current_points()
        if len(points['negative']) > 0:
            points['negative'].pop()
            self._set_status("Removed last negative point")
        elif len(points['positive']) > 0:
            points['positive'].pop()
            self._set_status("Removed last positive point")
        else:
            self._set_status("No points to remove")
            return

        self._update_segmentation()

    def _reset_points(self):
        """Reset all points on current frame"""
        if self.is_loading:
            return

        self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        self._set_status(f"Reset all points on frame {self.current_frame}")
        self._display_frame()

    def _save_and_quit(self):
        """Save results and quit"""
        if self.is_loading:
            return

        if len(self.video_segments) == 0:
            self._set_status("No frames to save!")
            return

        self.is_loading = True

        def save():
            try:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                masks_dir = output_dir / "masks"
                masks_dir.mkdir(exist_ok=True)

                total = len(self.video_segments)
                sorted_frames = sorted(self.video_segments.keys())

                # Save masks
                for i, frame_idx in enumerate(sorted_frames):
                    progress = int((i / total) * 50)
                    self._update_progress(f"Saving masks: {i+1}/{total}", progress)

                    masks = self.video_segments[frame_idx]
                    for obj_id, mask in masks.items():
                        mask_2d = mask.squeeze()
                        mask_img = (mask_2d * 255).astype(np.uint8)
                        cv2.imwrite(str(masks_dir / f"mask_{frame_idx:05d}.png"), mask_img)

                # Create video
                self._update_progress("Creating output video...", 55)
                first_frame = self._load_frame(sorted_frames[0])
                h, w = first_frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter('annotated_video.mp4', fourcc, self.fps, (w, h))

                for i, frame_idx in enumerate(sorted_frames):
                    progress = 55 + int((i / total) * 40)
                    self._update_progress(f"Writing video: {i+1}/{total}", progress)

                    frame = self._load_frame(frame_idx)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    masks = self.video_segments[frame_idx]

                    for obj_id, mask in masks.items():
                        mask_2d = mask.squeeze().astype(bool)
                        frame_bgr[mask_2d] = (frame_bgr[mask_2d] * 0.5 + np.array([255, 100, 0]) * 0.5).astype(np.uint8)

                    out_video.write(frame_bgr)

                out_video.release()

                self._update_progress(f"Saved {total} frames! Closing...", 100)
                self.root.after(1500, self.root.destroy)

            except Exception as e:
                self.is_loading = False
                self.root.after(0, lambda: self._set_status(f"Save error: {e}"))

        thread = threading.Thread(target=save, daemon=True)
        thread.start()

    def _quit(self):
        """Quit without saving"""
        if messagebox.askyesno("Quit", "Quit without saving?"):
            self.root.destroy()


def main():
    video_path = "input_4.mp4"
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return

    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(800, 600)

    app = VideoAnnotatorApp(root, video_path, checkpoint_path, model_cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
