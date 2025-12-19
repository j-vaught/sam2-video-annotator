#!/usr/bin/env python3
"""Create a demo video simulating water seepage for testing"""
import cv2
import numpy as np

def create_seepage_simulation(output_path="demo_seepage.mp4", duration_sec=10, fps=30):
    """Create a simple animation simulating water spreading"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    
    for frame_idx in range(num_frames):
        # Create base image (material)
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Simulate expanding water region
        t = frame_idx / num_frames
        center_x, center_y = width // 2, height // 3
        
        # Expanding circle with noise for irregular shape
        radius = int(50 + t * 150)
        
        # Create water mask with irregular edges
        mask = np.zeros((height, width), dtype=np.uint8)
        for angle in np.linspace(0, 2*np.pi, 100):
            noise = np.random.randint(-10, 10)
            r = radius + noise
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            cv2.circle(mask, (x, y), 5, 255, -1)
        
        # Fill the irregular shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Apply blue tint to wet areas
        img[mask > 0] = img[mask > 0] * 0.6 + np.array([150, 100, 50]) * 0.4
        
        # Add some texture
        noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add text
        cv2.putText(img, f"Simulated Seepage - Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(img)
    
    out.release()
    print(f"Created demo video: {output_path}")
    print(f"Duration: {duration_sec}s @ {fps} fps ({num_frames} frames)")

if __name__ == "__main__":
    create_seepage_simulation()
