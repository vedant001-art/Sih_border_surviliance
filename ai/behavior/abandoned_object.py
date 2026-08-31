from typing import List, Dict
import time
from loguru import logger
import math

class AbandonedObjectDetector:
    def __init__(self, stationary_threshold_seconds: float = 30.0, stationary_pixel_tolerance: float = 5.0):
        self.threshold = stationary_threshold_seconds
        self.pixel_tolerance = stationary_pixel_tolerance
        
        # Track IDs of backpacks/bags and their initial position + timestamp
        # track_id -> {"start_time": float, "last_pos": (cx, cy)}
        self.potential_abandoned = {}
        self.alerted_tracks = set()

    def update(self, detections: List[Dict], timestamp: float = None) -> List[Dict]:
        if timestamp is None:
            timestamp = time.time()
            
        events = []
        current_bags = set()
        
        for det in detections:
            if det["class_name"] not in ["backpack", "bag", "suitcase"]:
                continue
                
            track_id = det.get("track_id")
            if not track_id:
                continue
                
            current_bags.add(track_id)
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            if track_id not in self.potential_abandoned:
                self.potential_abandoned[track_id] = {"start_time": timestamp, "last_pos": (cx, cy)}
            else:
                # Check if it has moved
                last_cx, last_cy = self.potential_abandoned[track_id]["last_pos"]
                dist = math.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
                if dist > self.pixel_tolerance:
                    # Object moved, reset the timer
                    self.potential_abandoned[track_id] = {"start_time": timestamp, "last_pos": (cx, cy)}
                else:
                    # Object is stationary
                    duration = timestamp - self.potential_abandoned[track_id]["start_time"]
                    if duration > self.threshold and track_id not in self.alerted_tracks:
                        events.append({
                            "type": "ABANDONED_OBJECT",
                            "track_id": track_id,
                            "class_name": det["class_name"],
                            "message": f"Abandoned {det['class_name']} detected! Stationary for {int(duration)}s"
                        })
                        self.alerted_tracks.add(track_id)
                        
        # Cleanup objects that disappeared
        dead_tracks = [t for t in self.potential_abandoned.keys() if t not in current_bags]
        for t in dead_tracks:
            del self.potential_abandoned[t]
            if t in self.alerted_tracks:
                self.alerted_tracks.remove(t)
                
        return events
