import time
from typing import List, Dict

class LoiteringDetector:
    def __init__(self, threshold_seconds: float = 60.0):
        self.threshold_seconds = threshold_seconds
        # track_id -> {zone_name: start_time}
        self.track_zone_entry_times = {}
        # Track IDs that have already triggered an alert to avoid spam
        self.alerted_tracks = set()

    def update(self, current_intrusions: Dict[int, set], timestamp: float = None) -> List[Dict]:
        """
        current_intrusions: Mapping of track_id to set of zone names they are currently in.
        Returns a list of loitering events.
        """
        if timestamp is None:
            timestamp = time.time()
            
        events = []
        
        for track_id, zones in current_intrusions.items():
            if track_id not in self.track_zone_entry_times:
                self.track_zone_entry_times[track_id] = {}
                
            for zone in zones:
                if zone not in self.track_zone_entry_times[track_id]:
                    # newly entered
                    self.track_zone_entry_times[track_id][zone] = timestamp
                else:
                    # check duration
                    duration = timestamp - self.track_zone_entry_times[track_id][zone]
                    if duration > self.threshold_seconds and f"{track_id}_{zone}" not in self.alerted_tracks:
                        events.append({
                            "type": "LOITERING",
                            "track_id": track_id,
                            "zone_name": zone,
                            "duration": duration,
                            "message": f"Track #{track_id} loitering in {zone} for {int(duration)}s"
                        })
                        self.alerted_tracks.add(f"{track_id}_{zone}")
                        
        # Clean up tracks that are no longer in zones
        dead_tracks = []
        for track_id in list(self.track_zone_entry_times.keys()):
            if track_id not in current_intrusions:
                dead_tracks.append(track_id)
                continue
                
            # Remove zones they exited
            exited_zones = set(self.track_zone_entry_times[track_id].keys()) - current_intrusions[track_id]
            for z in exited_zones:
                del self.track_zone_entry_times[track_id][z]
                if f"{track_id}_{z}" in self.alerted_tracks:
                    self.alerted_tracks.remove(f"{track_id}_{z}")
                    
        for t in dead_tracks:
            del self.track_zone_entry_times[t]
            # Clean up alerted
            to_remove = [k for k in self.alerted_tracks if k.startswith(f"{t}_")]
            for k in to_remove:
                self.alerted_tracks.remove(k)
                
        return events
