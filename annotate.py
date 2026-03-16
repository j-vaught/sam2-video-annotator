#!/usr/bin/env python3
"""
Interactive Video Annotation with SAM2 - UNION MODE
Masks can only grow (union with previous frame) - perfect for fluid flow tracking
All SAM operations run in background threads for fluid GUI
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
import queue
import time

# Add SAM2 to path
sys.path.append('./segment-anything-2')
from sam2.build_sam import build_sam2_video_predictor


class VideoAnnotatorUnion:
    def __init__(self, root, video_path, checkpoint_path, model_cfg):
        self.root = root
        self.root.title("Video Annotation Tool - SAM2 [UNION MODE]")
        self.root.configure(bg='#2b2b2b')

        self.video_path = video_path
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.frame_dir = "frames"
        self.current_frame = 0

        # State
        self.video_segments = {}  # {frame_idx: {obj_id: mask}}
        self.frame_points = {}    # {frame_idx: {'positive': [], 'negative': []}}
        self.predictor = None
        self.inference_state = None
        self.total_frames = 0
        self.fps = 30
        self.display_scale = 1.0
        self.is_loading = False
        self.pending_update = False  # Flag for pending segmentation update
        self.stop_auto = False  # Flag to stop auto-propagation

        # Task queue for background processing
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_running = True
        self.worker_thread.start()

        # Setup UI
        self._setup_ui()

        # Start checking for results
        self._check_results()

        # Start initialization
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
        style.configure('Mode.TLabel', background='#1a472a', foreground='#90EE90', font=('Arial', 10, 'bold'))
        style.configure('TButton', font=('Arial', 10))

        # Top info bar
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=(0, 10))

        self.frame_label = ttk.Label(self.info_frame, text="Frame: 0/0", style='Title.TLabel')
        self.frame_label.pack(side=tk.LEFT)

        self.points_label = ttk.Label(self.info_frame, text="Points: +0 / -0", style='TLabel')
        self.points_label.pack(side=tk.LEFT, padx=20)

        # Union mode indicator
        self.mode_label = ttk.Label(self.info_frame, text=" UNION MODE ", style='Mode.TLabel')
        self.mode_label.pack(side=tk.LEFT, padx=10)

        self.area_label = ttk.Label(self.info_frame, text="", style='TLabel')
        self.area_label.pack(side=tk.RIGHT)

        # Canvas for video display
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self._on_left_click)
        self.canvas.bind('<Button-2>', self._on_right_click)
        self.canvas.bind('<Button-3>', self._on_right_click)

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

        ttk.Label(self.point_controls, text="Left=Add fluid, Right=Exclude",
                 style='Status.TLabel').pack(side=tk.LEFT)

        ttk.Button(self.point_controls, text="No Fluid (N)",
                  command=self._mark_no_fluid).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.point_controls, text="Clear Last (C)",
                  command=self._clear_last_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.point_controls, text="Reset Frame (R)",
                  command=self._reset_points).pack(side=tk.LEFT, padx=5)

        # Right side - Navigation
        self.nav_controls = ttk.Frame(self.button_frame)
        self.nav_controls.pack(side=tk.RIGHT)

        ttk.Button(self.nav_controls, text="Back (B)",
                  command=self._go_back).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_controls, text="Next (Space)",
                  command=self._next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_controls, text="Auto All (A)",
                  command=self._auto_propagate_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_controls, text="Save (S)",
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
        self.root.bind('<Escape>', lambda e: self._stop_auto())
        self.root.bind('x', lambda e: self._stop_auto())
        self.root.bind('b', lambda e: self._go_back())
        self.root.bind('a', lambda e: self._auto_propagate_all())
        self.root.bind('n', lambda e: self._mark_no_fluid())

    def _worker_loop(self):
        """Background worker thread for SAM operations"""
        while self.worker_running:
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    continue

                task_type = task['type']

                if task_type == 'init':
                    self._do_init(task)
                elif task_type == 'segment':
                    self._do_segment(task)
                elif task_type == 'propagate':
                    self._do_propagate(task)
                elif task_type == 'auto_propagate':
                    self._do_auto_propagate(task)
                elif task_type == 'save':
                    self._do_save(task)

            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put({'type': 'error', 'error': str(e)})

    def _check_results(self):
        """Check for results from worker thread (runs on main thread)"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_result(result)
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(50, self._check_results)

    def _handle_result(self, result):
        """Handle result from worker thread"""
        result_type = result.get('type')

        if result_type == 'progress':
            self.progress_label.config(text=result['message'])
            self.progress_bar['value'] = result['percent']
            self.status_label.config(text=result['message'])

        elif result_type == 'init_done':
            self.is_loading = False
            self._display_frame()

        elif result_type == 'segment_done':
            self.video_segments[result['frame']] = result['mask']
            self.is_loading = False
            self._display_frame()

        elif result_type == 'propagate_done':
            self.video_segments[result['frame']] = result['mask']
            self.current_frame = result['frame']
            if self.current_frame not in self.frame_points:
                self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
            self.is_loading = False
            self._display_frame()

        elif result_type == 'auto_frame_done':
            # Update during auto-propagation
            self.video_segments[result['frame']] = result['mask']
            self.current_frame = result['frame']
            if self.current_frame not in self.frame_points:
                self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
            self._display_frame()  # Update display but stay in loading mode

        elif result_type == 'auto_done':
            self.is_loading = False
            self.status_label.config(text=f"Done! Processed {result['count']} frames. Use B to review, S to save.")
            self._display_frame()

        elif result_type == 'auto_stopped':
            self.is_loading = False
            self.stop_auto = False
            self.status_label.config(text=f"Stopped after {result['count']} frames. Use B to go back and fix.")
            self._display_frame()

        elif result_type == 'save_done':
            self.root.after(1000, self.root.destroy)

        elif result_type == 'error':
            self.is_loading = False
            self.status_label.config(text=f"Error: {result['error']}")

    def _start_initialization(self):
        """Queue initialization task"""
        self.is_loading = True
        self.task_queue.put({'type': 'init'})

    def _do_init(self, task):
        """Initialize SAM2 (runs in worker thread)"""
        try:
            # Step 1: Extract frames
            self.result_queue.put({'type': 'progress', 'message': 'Extracting video frames...', 'percent': 0})
            self._extract_frames()

            # Step 2: Load SAM2
            self.result_queue.put({'type': 'progress', 'message': 'Loading SAM2 model...', 'percent': 33})
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=device)

            # Step 3: Initialize video state
            self.result_queue.put({'type': 'progress', 'message': 'Initializing video state...', 'percent': 66})
            self.inference_state = self.predictor.init_state(
                video_path=str(Path(self.frame_dir).absolute())
            )

            # Done
            self.result_queue.put({'type': 'progress', 'message': 'Ready! Click to add points, Space for next frame.', 'percent': 100})
            self.result_queue.put({'type': 'init_done'})

        except Exception as e:
            self.result_queue.put({'type': 'error', 'error': str(e)})

    def _extract_frames(self):
        """Extract frames from video"""
        output_path = Path(self.frame_dir)
        output_path.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        existing_frames = list(output_path.glob("*.jpg"))
        if len(existing_frames) >= self.total_frames:
            cap.release()
            return

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
                self.result_queue.put({'type': 'progress', 'message': f'Extracting: {frame_idx}/{self.total_frames}', 'percent': progress})

        cap.release()

    def _load_frame(self, frame_idx):
        """Load a frame as numpy array (RGB)"""
        frame_path = Path(self.frame_dir) / f"{frame_idx:05d}.jpg"
        if not frame_path.exists():
            return None
        frame = cv2.imread(str(frame_path))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _display_frame(self):
        """Display current frame with mask and points"""
        if self.is_loading and not self.video_segments:
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
                # Green overlay for fluid (union mode)
                overlay_color = np.array([100, 255, 150])
                display[mask_bool] = (display[mask_bool] * 0.5 + overlay_color * 0.5).astype(np.uint8)

        # Draw points
        points = self._get_current_points()
        for pt in points['positive']:
            cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
            cv2.circle(display, tuple(pt), 9, (255, 255, 255), 2)
        for pt in points['negative']:
            cv2.circle(display, tuple(pt), 8, (255, 0, 0), -1)
            cv2.circle(display, tuple(pt), 9, (255, 255, 255), 2)

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

        # Show loading indicator
        if self.is_loading:
            self.status_label.config(text="Processing...")

    def _get_current_points(self):
        """Get points for current frame"""
        if self.current_frame not in self.frame_points:
            self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        return self.frame_points[self.current_frame]

    def _canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        frame = self._load_frame(self.current_frame)
        if frame is None:
            return None, None

        h, w = frame.shape[:2]
        disp_w = int(w * self.display_scale)
        disp_h = int(h * self.display_scale)

        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2

        img_x = int((x - offset_x) / self.display_scale)
        img_y = int((y - offset_y) / self.display_scale)

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
        self.status_label.config(text=f"Added point at ({img_x}, {img_y}) - processing...")
        self._queue_segmentation()

    def _on_right_click(self, event):
        """Handle right click - add negative point"""
        if self.is_loading:
            return

        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        if img_x is None:
            return

        points = self._get_current_points()
        points['negative'].append([img_x, img_y])
        self.status_label.config(text=f"Added exclusion at ({img_x}, {img_y}) - processing...")
        self._queue_segmentation()

    def _queue_segmentation(self):
        """Queue segmentation update"""
        points = self._get_current_points()
        if len(points['positive']) == 0:
            self._display_frame()
            return

        # Clear future frames since we're modifying this one
        cleared = self._clear_future_frames(self.current_frame)
        if cleared > 0:
            self.status_label.config(text=f"Cleared {cleared} future frames. Processing...")

        self.is_loading = True
        self._display_frame()  # Show loading state

        # Get previous mask for union
        prev_mask = None
        if self.current_frame > 0 and (self.current_frame - 1) in self.video_segments:
            prev_mask = self.video_segments[self.current_frame - 1][1].copy()

        self.task_queue.put({
            'type': 'segment',
            'frame': self.current_frame,
            'points': points.copy(),
            'prev_mask': prev_mask
        })

    def _do_segment(self, task):
        """Run segmentation (in worker thread)"""
        try:
            frame_idx = task['frame']
            points = task['points']
            prev_mask = task['prev_mask']

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
                frame_idx=frame_idx,
                obj_id=1,
                points=all_points,
                labels=labels,
            )

            mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            # UNION MODE: Combine with previous mask
            if prev_mask is not None:
                mask = np.logical_or(mask, prev_mask)

            self.result_queue.put({
                'type': 'segment_done',
                'frame': frame_idx,
                'mask': {1: mask}
            })

        except Exception as e:
            self.result_queue.put({'type': 'error', 'error': str(e)})

    def _next_frame(self):
        """Propagate to next frame"""
        if self.is_loading:
            return

        points = self._get_current_points()
        if not points['positive'] and self.current_frame not in self.video_segments:
            self.status_label.config(text="Add at least one positive point first (or press N for No Fluid)!")
            return

        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.status_label.config(text="Reached end of video! Press S to save.")
            return

        # Check if current frame has an empty mask (no fluid)
        current_mask = None
        is_empty_mask = False
        if self.current_frame in self.video_segments:
            current_mask = self.video_segments[self.current_frame][1].copy()
            is_empty_mask = not np.any(current_mask)

        # If current mask is empty (no fluid), just move to next frame without SAM propagation
        if is_empty_mask:
            self.current_frame = next_frame
            if next_frame not in self.frame_points:
                self.frame_points[next_frame] = {'positive': [], 'negative': []}
            self.status_label.config(text=f"Moved to frame {next_frame}. Add points for fluid or press N for No Fluid.")
            self._display_frame()
            return

        # Otherwise, propagate using SAM
        self.is_loading = True
        self.status_label.config(text=f"Propagating to frame {next_frame}...")
        self.progress_bar['value'] = 50

        self.task_queue.put({
            'type': 'propagate',
            'next_frame': next_frame,
            'current_mask': current_mask
        })

    def _do_propagate(self, task):
        """Run propagation (in worker thread)"""
        try:
            next_frame = task['next_frame']
            current_mask = task['current_mask']

            self.result_queue.put({'type': 'progress', 'message': f'Propagating to frame {next_frame}...', 'percent': 50})

            # Run SAM2 propagation
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=next_frame, max_frame_num_to_track=1
            ):
                if out_frame_idx == next_frame:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()

                    # UNION MODE: New mask must include all of previous mask
                    if current_mask is not None:
                        mask = np.logical_or(mask, current_mask)

                    self.result_queue.put({
                        'type': 'propagate_done',
                        'frame': next_frame,
                        'mask': {1: mask}
                    })
                    self.result_queue.put({'type': 'progress', 'message': f'Frame {next_frame} ready', 'percent': 100})
                    break

        except Exception as e:
            self.result_queue.put({'type': 'error', 'error': str(e)})

    def _stop_auto(self):
        """Stop auto-propagation"""
        if self.is_loading:
            self.stop_auto = True
            self.status_label.config(text="Stopping... (will stop after current frame)")

    def _auto_propagate_all(self):
        """Auto-propagate through all remaining frames"""
        if self.is_loading:
            return

        # Must have at least one frame segmented
        if self.current_frame not in self.video_segments:
            points = self._get_current_points()
            if not points['positive']:
                self.status_label.config(text="Add points to current frame first!")
                return

        remaining = self.total_frames - self.current_frame - 1
        if remaining <= 0:
            self.status_label.config(text="Already at last frame!")
            return

        self.is_loading = True
        self.stop_auto = False  # Reset stop flag
        self.status_label.config(text=f"Auto-propagating {remaining} frames... (Esc/X to stop)")

        # Get current mask for union
        current_mask = None
        if self.current_frame in self.video_segments:
            current_mask = self.video_segments[self.current_frame][1].copy()

        self.task_queue.put({
            'type': 'auto_propagate',
            'start_frame': self.current_frame + 1,
            'end_frame': self.total_frames,
            'current_mask': current_mask
        })

    def _do_auto_propagate(self, task):
        """Auto-propagate through all frames (in worker thread)"""
        try:
            start_frame = task['start_frame']
            end_frame = task['end_frame']
            prev_mask = task['current_mask']

            total = end_frame - start_frame
            processed = 0

            for frame_idx in range(start_frame, end_frame):
                # Check if stop requested
                if self.stop_auto:
                    self.result_queue.put({'type': 'progress', 'message': f'Stopped at frame {frame_idx}', 'percent': 100})
                    self.result_queue.put({'type': 'auto_stopped', 'count': processed, 'stopped_at': frame_idx})
                    return

                # Update progress
                progress = int((processed / total) * 100)
                self.result_queue.put({
                    'type': 'progress',
                    'message': f'Auto: {processed}/{total} (frame {frame_idx}) - Esc to stop',
                    'percent': progress
                })

                # Propagate to this frame
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=frame_idx, max_frame_num_to_track=1
                ):
                    if out_frame_idx == frame_idx:
                        mask = (out_mask_logits[0] > 0.0).cpu().numpy()

                        # UNION MODE: Combine with previous mask
                        if prev_mask is not None:
                            mask = np.logical_or(mask, prev_mask)

                        # Store for next iteration
                        prev_mask = mask.copy()

                        # Send result to update display
                        self.result_queue.put({
                            'type': 'auto_frame_done',
                            'frame': frame_idx,
                            'mask': {1: mask}
                        })
                        break

                processed += 1

            # Done
            self.result_queue.put({'type': 'progress', 'message': 'Auto-propagation complete!', 'percent': 100})
            self.result_queue.put({'type': 'auto_done', 'count': processed})

        except Exception as e:
            self.result_queue.put({'type': 'error', 'error': str(e)})

    def _clear_future_frames(self, from_frame):
        """Clear all frames after from_frame (they're now invalid in union mode)"""
        frames_to_remove = [f for f in self.video_segments.keys() if f > from_frame]
        for f in frames_to_remove:
            del self.video_segments[f]
            if f in self.frame_points:
                del self.frame_points[f]
        if frames_to_remove:
            return len(frames_to_remove)
        return 0

    def _go_back(self):
        """Go back to previous frame"""
        if self.is_loading:
            return

        if self.current_frame <= 0:
            self.status_label.config(text="Already at first frame")
            return

        prev_frame = self.current_frame - 1
        if prev_frame in self.video_segments:
            self.current_frame = prev_frame
            self.status_label.config(text=f"Went back to frame {prev_frame}. Modify here, then Space to re-propagate.")
            self._display_frame()
        else:
            self.status_label.config(text=f"Frame {prev_frame} not processed yet")

    def _clear_last_point(self):
        """Remove last added point"""
        if self.is_loading:
            return

        points = self._get_current_points()
        if len(points['negative']) > 0:
            points['negative'].pop()
            self.status_label.config(text="Removed last exclusion point")
        elif len(points['positive']) > 0:
            points['positive'].pop()
            self.status_label.config(text="Removed last point")
        else:
            self.status_label.config(text="No points to remove")
            return

        self._queue_segmentation()

    def _reset_points(self):
        """Reset all points on current frame"""
        if self.is_loading:
            return

        # Clear future frames since we're modifying this one
        cleared = self._clear_future_frames(self.current_frame)

        self.frame_points[self.current_frame] = {'positive': [], 'negative': []}
        # Also remove this frame's mask
        if self.current_frame in self.video_segments:
            del self.video_segments[self.current_frame]

        msg = f"Reset frame {self.current_frame}"
        if cleared > 0:
            msg += f" (cleared {cleared} future frames)"
        self.status_label.config(text=msg)
        self._display_frame()

    def _mark_no_fluid(self):
        """Mark current frame as having no fluid (empty mask)"""
        if self.is_loading:
            return

        # Clear future frames since we're modifying this one
        cleared = self._clear_future_frames(self.current_frame)

        # Get frame dimensions to create properly sized empty mask
        frame = self._load_frame(self.current_frame)
        if frame is None:
            return

        h, w = frame.shape[:2]

        # Create empty mask (all False)
        empty_mask = np.zeros((1, h, w), dtype=bool)

        # Store the empty mask
        self.video_segments[self.current_frame] = {1: empty_mask}

        # Clear any points on this frame
        self.frame_points[self.current_frame] = {'positive': [], 'negative': []}

        msg = f"Frame {self.current_frame} = NO FLUID"
        if cleared > 0:
            msg += f" (cleared {cleared} future frames)"
        self.status_label.config(text=msg)
        self._display_frame()

    def _save_and_quit(self):
        """Save results and quit"""
        if self.is_loading:
            return

        if len(self.video_segments) == 0:
            self.status_label.config(text="No frames to save!")
            return

        self.is_loading = True
        self.task_queue.put({'type': 'save'})

    def _do_save(self, task):
        """Save results (in worker thread)"""
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
                self.result_queue.put({'type': 'progress', 'message': f'Saving masks: {i+1}/{total}', 'percent': progress})

                masks = self.video_segments[frame_idx]
                for obj_id, mask in masks.items():
                    mask_2d = mask.squeeze()
                    mask_img = (mask_2d * 255).astype(np.uint8)
                    cv2.imwrite(str(masks_dir / f"mask_{frame_idx:05d}.png"), mask_img)

            # Create video
            self.result_queue.put({'type': 'progress', 'message': 'Creating video...', 'percent': 55})

            first_frame = self._load_frame(sorted_frames[0])
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter('annotated_video_union.mp4', fourcc, self.fps, (w, h))

            for i, frame_idx in enumerate(sorted_frames):
                progress = 55 + int((i / total) * 40)
                self.result_queue.put({'type': 'progress', 'message': f'Writing: {i+1}/{total}', 'percent': progress})

                frame = self._load_frame(frame_idx)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                masks = self.video_segments[frame_idx]

                for obj_id, mask in masks.items():
                    mask_2d = mask.squeeze().astype(bool)
                    # Green overlay for union mode
                    frame_bgr[mask_2d] = (frame_bgr[mask_2d] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)

                out_video.write(frame_bgr)

            out_video.release()

            self.result_queue.put({'type': 'progress', 'message': f'Saved {total} frames!', 'percent': 100})
            self.result_queue.put({'type': 'save_done'})

        except Exception as e:
            self.result_queue.put({'type': 'error', 'error': str(e)})

    def _quit(self):
        """Quit without saving"""
        if messagebox.askyesno("Quit", "Quit without saving?"):
            self.worker_running = False
            self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Video Annotation with SAM2 - Union Mode')
    parser.add_argument('--video', '-v', default='input_4.mp4', help='Video file path')
    parser.add_argument('--model', '-m', default='large', choices=['tiny', 'small', 'large'],
                       help='SAM2 model size: tiny (fastest), small, large (most accurate)')
    args = parser.parse_args()

    video_path = args.video

    # Model configurations
    models = {
        'tiny': ('checkpoints/sam2.1_hiera_tiny.pt', 'configs/sam2.1/sam2.1_hiera_t.yaml'),
        'small': ('checkpoints/sam2.1_hiera_small.pt', 'configs/sam2.1/sam2.1_hiera_s.yaml'),
        'large': ('checkpoints/sam2.1_hiera_large.pt', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
    }

    checkpoint_path, model_cfg = models[args.model]
    print(f"Using SAM2 model: {args.model}")

    # Check if checkpoint exists, if not try to download or use large
    if not Path(checkpoint_path).exists():
        print(f"Warning: {checkpoint_path} not found, falling back to large model")
        checkpoint_path, model_cfg = models['large']

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return

    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(800, 600)

    app = VideoAnnotatorUnion(root, video_path, checkpoint_path, model_cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
