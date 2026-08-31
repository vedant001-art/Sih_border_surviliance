import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
from typing import List, Dict, Tuple
from loguru import logger

class VirtualFence:
    def __init__(self, zones: List[Dict]):
        """
        zones: [
            {"id": 1, "name": "Restricted Zone A", "type": "POLYGON", "coords": [(x,y), (x,y), ...]},
            {"id": 2, "name": "Tripwire", "type": "LINE", "coords": [(x1,y1), (x2,y2)]}
        ]
        """
        self.polygons = {}
        self.lines = {}
        
        for z in zones:
            coords = z.get("coords", [])
            if z.get("type") == "POLYGON" and len(coords) >= 3:
                try:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    self.polygons[z["name"]] = poly
                except Exception as e:
                    logger.warning(f"Failed to create polygon for zone {z.get('name')}: {e}")
            elif z.get("type") == "LINE" and len(coords) >= 2:
                try:
                    self.lines[z["name"]] = LineString(coords)
                except Exception as e:
                    logger.warning(f"Failed to create tripwire for zone {z.get('name')}: {e}")
                
        # Track state to avoid spamming alerts
        # track_id -> set of zones they are currently inside
        self.active_intrusions = {}
        self.alert_cache = {}

    def check_intrusion(self, detections: List[Dict]) -> List[Dict]:
        """
        Checks if detections breach the virtual fence boundary.
        Evaluates centroid, bottom ground-contact point (feet/wheels), and full bounding box intersection.
        Returns a list of intrusion events.
        """
        events = []
        current_frame_intrusions = {}
        
        for det in detections:
            if "track_id" not in det or "bbox" not in det:
                continue
                
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            p_center = Point(cx, cy)
            p_bottom = Point(cx, y2)
            p_top = Point(cx, y1)
            target_box = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            
            in_zones = set()
            
            # Check restricted polygons
            for zone_name, poly in self.polygons.items():
                try:
                    p = poly if poly.is_valid else poly.buffer(0)
                    if (p.contains(p_center) or 
                        p.contains(p_bottom) or 
                        p.contains(p_top) or 
                        p.intersects(target_box)):
                        in_zones.add(zone_name)
                except Exception:
                    pass
            
            # Check tripwire lines
            for line_name, line in self.lines.items():
                try:
                    if line.intersects(target_box):
                        in_zones.add(line_name)
                except Exception:
                    pass
                    
            current_frame_intrusions[track_id] = in_zones
            
            for z in in_zones:
                events.append({
                    "type": "INTRUSION",
                    "track_id": track_id,
                    "class_name": det.get("class_name", "target"),
                    "zone_name": z,
                    "message": f"{det.get('class_name', 'target').upper()} #{track_id} entered {z}"
                })
                    
        self.active_intrusions = current_frame_intrusions
        return events

    def check_approaching(self, detections: List[Dict], motion_tracker=None) -> List[Dict]:
        """
        Predictive Trajectory Analysis:
        Checks if moving targets (vehicles/personnel) have a motion trajectory vector or proximity
        that is approaching a restricted Virtual Fence polygon.
        Returns a list of APPROACHING warning events.
        """
        events = []
        
        for det in detections:
            if "track_id" not in det or "bbox" not in det:
                continue
                
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            p_bottom = Point(cx, y2)
            p_center = Point(cx, cy)
            
            hist = motion_tracker.get_history(track_id) if motion_tracker else []
            if len(hist) < 3:
                continue
                
            # Recent velocity vector over last 3-5 frames
            dx = hist[-1][0] - hist[-3][0] if len(hist) >= 3 else 0.0
            dy = hist[-1][1] - hist[-3][1] if len(hist) >= 3 else 0.0
            movement = np.hypot(dx, dy)
            
            # Target must be actively moving (movement > 1.2px per frame)
            if movement < 1.2:
                continue
                
            # Project trajectory 2 seconds into future
            p_future = Point(cx + dx * 3.5, y2 + dy * 3.5)
            
            for zone_name, poly in self.polygons.items():
                try:
                    p = poly if poly.is_valid else poly.buffer(0)
                    
                    # Target is ALREADY inside -> skip approaching event (already intrusion)
                    if p.contains(p_center) or p.contains(p_bottom):
                        continue
                        
                    # Expanded 50-pixel Proximity Early Warning Buffer Zone
                    buffer_zone = p.buffer(55)
                    
                    # Approaching if projected future trajectory enters poly OR ground contact point enters proximity buffer
                    if p.contains(p_future) or buffer_zone.contains(p_bottom):
                        events.append({
                            "type": "APPROACHING",
                            "track_id": track_id,
                            "class_name": det.get("class_name", "target"),
                            "zone_name": zone_name,
                            "message": f"WARNING: Trajectory of {det.get('class_name', 'target').upper()} #{track_id} is approaching {zone_name}."
                        })
                except Exception:
                    pass
                    
        return events
