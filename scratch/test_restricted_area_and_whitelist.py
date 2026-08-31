import sys
import os
import time
import numpy as np

BASE_DIR = r"c:\Users\lenovo\OneDrive\Desktop\sih"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai.pipeline import CameraPipeline
from ai.behavior.virtual_fence import VirtualFence
from backend.services.authorized_plates import authorized_plates

print("--- 1. Testing Authorized Plates Flexible Matcher ---")
print("Registered plates:", authorized_plates.get_all())
assert authorized_plates.is_authorized("DL8CAF1234") is True
assert authorized_plates.is_authorized("DL 8C AF 1234") is True
assert authorized_plates.is_authorized("3789") is True
assert authorized_plates.is_authorized("KA013789") is True  # Substring match
assert authorized_plates.is_authorized("DLBCAF1234") is True # OCR 8 vs B canonicalization
assert authorized_plates.is_authorized("UNKNOWN_CAR_999") is False
print(">>> Authorized plate matcher verified successfully!")

print("\n--- 2. Testing Alert Suppression inside Restricted Area ---")
pipe = CameraPipeline("TEST-RESTRICTED")

# Define a restricted area polygon: [200, 200] to [500, 500]
pipe.virtual_fence = VirtualFence(zones=[{
    "id": 1,
    "name": "Restricted Zone Sector A",
    "type": "POLYGON",
    "coords": [(200, 200), (500, 200), (500, 500), (200, 500)]
}])

dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)

# Target 1: Car inside restricted area WITH AUTHORIZED PLATE
det_whitelisted = {
    "track_id": 1,
    "bbox": [250, 250, 350, 350],  # Centroid at (300, 300) -> INSIDE
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.85
}
pipe._plate_cache[1] = {"plate": "DL8CAF1234"}

# Target 2: Car OUTSIDE restricted area (Normal traffic on road)
det_outside = {
    "track_id": 2,
    "bbox": [50, 50, 150, 150],    # Centroid at (100, 100) -> OUTSIDE
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.85
}
pipe._plate_cache[2] = {"plate": "UNREGISTERED_ROAD_CAR"}

# Target 3: Car inside restricted area WITH UNREGISTERED PLATE
det_unauthorized = {
    "track_id": 3,
    "bbox": [300, 300, 450, 450],  # Centroid at (375, 375) -> INSIDE
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.85
}
pipe._plate_cache[3] = {"plate": "UNAUTHORIZED_9999"}

# Track alerts sent to DB worker
intercepted_alerts = []
def mock_enqueue(task, data):
    if task == "CREATE_EVENT":
        intercepted_alerts.append(data)

from backend.services.db_worker import db_worker
db_worker.enqueue_task = mock_enqueue

# Run evaluation
pipe._evaluate_rules([det_whitelisted, det_outside, det_unauthorized], dummy_frame)

print(f"Total alerts intercepted: {len(intercepted_alerts)}")
for a in intercepted_alerts:
    print(f"  -> Generated Alert: {a.get('title')}")

# Verify assertions:
# 1. Car 1 (whitelisted) MUST NOT trigger any alert
car_1_alerts = [a for a in intercepted_alerts if a.get("local_track_id") == 1]
assert len(car_1_alerts) == 0, f"Car #1 (whitelisted) triggered alerts: {car_1_alerts}"
print("✓ Car #1 (Whitelisted) inside restricted zone triggered ZERO alerts (Suppressed).")

# 2. Car 2 (outside restricted area) MUST NOT trigger any alert
car_2_alerts = [a for a in intercepted_alerts if a.get("local_track_id") == 2]
assert len(car_2_alerts) == 0, f"Car #2 (outside zone) triggered alerts: {car_2_alerts}"
print("✓ Car #2 (Outside restricted zone) triggered ZERO alerts.")

# 3. Car 3 (unauthorized inside restricted area) MUST trigger alert
car_3_alerts = [a for a in intercepted_alerts if a.get("local_track_id") == 3]
assert len(car_3_alerts) == 1, f"Expected 1 alert for unauthorized car, got {len(car_3_alerts)}"
print(f"✓ Car #3 (Unauthorized inside restricted zone) triggered alert: {car_3_alerts[0]['title']}")

print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
