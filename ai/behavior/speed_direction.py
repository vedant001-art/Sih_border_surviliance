import math
from typing import Dict, List, Tuple
import time

class SpeedDirectionAnalyzer:
    def __init__(self, fps: float = 30.0, pixel_to_meter: float = 0.05):
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        # track_id -> [(timestamp, (cx, cy)), ...]
        self.history: Dict[int, List[Tuple[float, Tuple[float, float]]]] = {}
        self.history_max_len = 30

    def update(self, detections: List[Dict], timestamp: float = None) -> List[Dict]:
        """
        Updates the history for each track and calculates speed and direction.
        Returns the detections augmented with 'speed_kmh' and 'direction'.
        """
        if timestamp is None:
            timestamp = time.time()
            
        current_tracks = set()
        
        for det in detections:
            if "track_id" not in det:
                continue
                
            track_id = det["track_id"]
            current_tracks.add(track_id)
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            if track_id not in self.history:
                self.history[track_id] = []
            
            self.history[track_id].append((timestamp, (cx, cy)))
            if len(self.history[track_id]) > self.history_max_len:
                self.history[track_id].pop(0)
                
            # Calculate Speed and Direction
            if len(self.history[track_id]) >= 5:
                # Compare current with 5 frames ago to smooth out jitter
                t_old, (cx_old, cy_old) = self.history[track_id][-5]
                dt = timestamp - t_old
                
                if dt > 0:
                    dx = cx - cx_old
                    dy = cy - cy_old
                    dist_pixels = math.sqrt(dx**2 + dy**2)
                    dist_meters = dist_pixels * self.pixel_to_meter
                    speed_mps = dist_meters / dt
                    speed_kmh = speed_mps * 3.6
                    det["speed_kmh"] = round(speed_kmh, 1)
                    
                    # Direction
                    angle = math.degrees(math.atan2(-dy, dx)) # -dy because image y goes down
                    if -45 <= angle < 45:
                        direction = "East"
                    elif 45 <= angle < 135:
                        direction = "North"
                    elif angle >= 135 or angle < -135:
                        direction = "West"
                    else:
                        direction = "South"
                        
                    det["direction"] = direction
                else:
                    det["speed_kmh"] = 0.0
                    det["direction"] = "Unknown"
            else:
                det["speed_kmh"] = 0.0
                det["direction"] = "Unknown"
                
        # Cleanup
        dead_tracks = [t for t in self.history.keys() if t not in current_tracks]
        for t in dead_tracks:
            # We don't delete immediately if we want to detect abandoned objects, but for speed we can.
            # Wait, keep it simple for now.
            del self.history[t]
            
        return detections
