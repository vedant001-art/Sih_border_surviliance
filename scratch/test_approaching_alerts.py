import sys
import os
sys.path.append(os.getcwd())

from ai.behavior.virtual_fence import VirtualFence
from backend.services.tracker import TrackMemory

def test_approaching_trajectory():
    print("=== Testing Predictive Trajectory Approaching Alerts ===")
    
    # Define a polygon restricted zone (Restricted Sector CAM-01)
    zones = [{
        "id": 1,
        "name": "Restricted Sector CAM-01",
        "type": "POLYGON",
        "coords": [(200, 200), (500, 200), (500, 500), (200, 500)]
    }]
    
    fence = VirtualFence(zones=zones)
    tracker = TrackMemory()
    
    # Target vehicle outside the polygon, moving towards it
    # Frame 1: at (100, 350)
    tracker.update(local_track_id=10, frame_idx=1, center_x=100.0, center_y=350.0)
    # Frame 2: at (130, 350)
    tracker.update(local_track_id=10, frame_idx=2, center_x=130.0, center_y=350.0)
    # Frame 3: at (160, 350) - trajectory vector dx=+30, dy=0 points straight into polygon at x=200!
    tracker.update(local_track_id=10, frame_idx=3, center_x=160.0, center_y=350.0)
    
    detections = [{
        "track_id": 10,
        "bbox": [140.0, 330.0, 180.0, 370.0],
        "class_name": "car",
        "speed_kmh": 25.0
    }]
    
    intrusions = fence.check_intrusion(detections)
    approaching = fence.check_approaching(detections, motion_tracker=tracker)
    
    print(f"Intrusions count (should be 0 because target is outside): {len(intrusions)}")
    print(f"Approaching warnings count (should be 1): {len(approaching)}")
    
    if len(approaching) > 0:
        evt = approaching[0]
        print(f"  - Type: {evt['type']}")
        print(f"  - Track: #{evt['track_id']}")
        print(f"  - Zone: {evt['zone_name']}")
        print(f"  - Message: {evt['message']}")
        assert evt["type"] == "APPROACHING"
        assert evt["track_id"] == 10
        print(">>> PREDICTIVE TRAJECTORY APPROACHING ALERT TEST PASSED 100%! <<<")
    else:
        print("!!! APPROACHING TEST FAILED !!!")

if __name__ == "__main__":
    test_approaching_trajectory()
