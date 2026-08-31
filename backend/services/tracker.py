import math
import time
from typing import Dict, List, Tuple
from collections import deque

class TrackMemory:
    def __init__(self, buffer_frames=90):
        self.buffer_frames = buffer_frames
        # local_track_id -> list of [x, y] history
        self.history: Dict[int, deque] = {}
        # local_track_id -> last known frame idx
        self.last_seen_frame: Dict[int, int] = {}
        
    def update(self, local_track_id: int, frame_idx: int, center_x: float, center_y: float):
        if local_track_id not in self.history:
            self.history[local_track_id] = deque(maxlen=self.buffer_frames)
        self.history[local_track_id].append((center_x, center_y))
        self.last_seen_frame[local_track_id] = frame_idx
        
    def get_velocity(self, local_track_id: int) -> Tuple[float, float]:
        """Returns rough velocity vector [vx, vy] over the recent frames"""
        hist = self.history.get(local_track_id, [])
        if len(hist) < 5:
            return (0.0, 0.0)
            
        dx = hist[-1][0] - hist[-5][0]
        dy = hist[-1][1] - hist[-5][1]
        return (dx, dy)
        
    def get_speed_kmh(self, local_track_id: int, fps=30.0, meters_per_pixel=0.02) -> float:
        vx, vy = self.get_velocity(local_track_id)
        # pixels per 5 frames -> pixels per frame
        px_per_frame = math.sqrt(vx**2 + vy**2) / 5.0
        # px/frame * frames/sec * meters/px -> meters/sec
        mps = px_per_frame * fps * meters_per_pixel
        kmh = mps * 3.6
        return min(kmh, 150.0) # Cap at 150 km/h for sanity
        
    def get_heading(self, local_track_id: int) -> str:
        vx, vy = self.get_velocity(local_track_id)
        if abs(vx) < 2 and abs(vy) < 2:
            return "Stationary"
            
        if abs(vx) > abs(vy):
            return "Eastbound" if vx > 0 else "Westbound"
        else:
            return "Southbound" if vy > 0 else "Northbound"

    def is_recently_lost(self, local_track_id: int, current_frame: int) -> bool:
        last = self.last_seen_frame.get(local_track_id, 0)
        return 0 < (current_frame - last) < self.buffer_frames
        
    def get_history(self, local_track_id: int) -> list:
        return list(self.history.get(local_track_id, []))
        
    def cleanup(self, current_frame: int):
        expired = [tid for tid, last in self.last_seen_frame.items() if current_frame - last > self.buffer_frames * 2]
        for tid in expired:
            del self.last_seen_frame[tid]
            if tid in self.history:
                del self.history[tid]

# Global instance for use in pipeline
motion_tracker = TrackMemory()
