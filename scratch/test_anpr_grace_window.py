import sys
import os
import time
import numpy as np

BASE_DIR = r"c:\Users\lenovo\OneDrive\Desktop\sih"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai.pipeline import CameraPipeline
from ai.behavior.virtual_fence import VirtualFence

pipe = CameraPipeline("TEST-GRACE-PERIOD")
pipe.virtual_fence = VirtualFence(zones=[{
    "id": 1,
    "name": "Restricted Sector",
    "type": "POLYGON",
    "coords": [(100, 100), (500, 100), (500, 500), (100, 500)]
}])

dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)

intercepted_alerts = []
def mock_enqueue(task, data):
    if task == "CREATE_EVENT":
        intercepted_alerts.append(data)

from backend.services.db_worker import db_worker
db_worker.enqueue_task = mock_enqueue

det_car = {
    "track_id": 10,
    "bbox": [150, 150, 300, 300],
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.9
}

print("--- Frame 0: Car enters restricted zone (Plate not yet scanned) ---")
pipe._evaluate_rules([det_car], dummy_frame)
assert len(intercepted_alerts) == 0, f"Alert fired prematurely: {intercepted_alerts}"
print("✓ No premature alert fired! (Grace window active waiting for ANPR OCR)")

print("--- Frame 5: ANPR OCR finishes and recognizes authorized plate DL8CAF1234 ---")
pipe._plate_cache[10] = {"plate": "DL8CAF1234"}
pipe._evaluate_rules([det_car], dummy_frame)
assert len(intercepted_alerts) == 0, f"Alert fired for whitelisted car: {intercepted_alerts}"
assert 10 in pipe.authorized_tracks
print("✓ Car #10 recognized as AUTHORIZED; Alert SUPPRESSED! Zero alerts fired.")

print("--- Next Frames: Car continues inside restricted zone ---")
time.sleep(2.1)
pipe._evaluate_rules([det_car], dummy_frame)
assert len(intercepted_alerts) == 0, f"Alert fired on subsequent frames: {intercepted_alerts}"
print("✓ Verified: Even after 2+ seconds inside zone, whitelisted car NEVER triggers alerts.")

print("\n>>> ANPR GRACE WINDOW & WHITELIST PERSISTENCE PASSED! <<<")
