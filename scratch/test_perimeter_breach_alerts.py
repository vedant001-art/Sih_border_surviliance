import sys
sys.path.insert(0, ".")
import time
from fastapi.testclient import TestClient
from backend.main import app
from ai.behavior.virtual_fence import VirtualFence
from shapely.geometry import box, Point

# 1. Unit Test: Virtual Fence Intersection
print("=== 1. Testing Virtual Fence Intersection Engine ===")
fence = VirtualFence(zones=[{
    "id": 1,
    "name": "Highway Restricted Corridor",
    "type": "POLYGON",
    "coords": [(100, 300), (500, 300), (500, 450), (100, 450)]
}])

# Test A: Target whose centroid is outside, but wheels/feet touch fence
det_touching = {"track_id": 101, "class_name": "car", "bbox": [200, 200, 350, 320]}
res_touching = fence.check_intrusion([det_touching])
assert len(res_touching) == 1, f"Expected 1 intrusion, got {len(res_touching)}"
print("✓ Target with bottom touching fence detected successfully!")

# Test B: Fast moving vehicle bounding box overlapping corridor
det_fast = {"track_id": 102, "class_name": "truck", "bbox": [150, 280, 400, 480]}
res_fast = fence.check_intrusion([det_fast])
assert len(res_fast) == 1, f"Expected 1 intrusion, got {len(res_fast)}"
print("✓ Fast vehicle overlapping corridor detected successfully!")

# Test C: Pedestrian whose feet walk into fence
det_person = {"track_id": 103, "class_name": "person", "bbox": [250, 220, 300, 310]}
res_person = fence.check_intrusion([det_person])
assert len(res_person) == 1, f"Expected 1 intrusion, got {len(res_person)}"
print("✓ Pedestrian crossing into fence detected successfully!")

# Test D: Target completely outside
det_outside = {"track_id": 104, "class_name": "car", "bbox": [600, 100, 800, 250]}
res_outside = fence.check_intrusion([det_outside])
assert len(res_outside) == 0, f"Expected 0 intrusions, got {len(res_outside)}"
print("✓ Target outside fence correctly ignored.")

# 2. Integration Test: API and Alert Retrieval for Perimeter Panel
print("\n=== 2. Testing API & Alert Retrieval for Perimeter Panel ===")
from backend.services.db_worker import db_worker
db_worker.start()
client = TestClient(app)

# Reset DB for test
r = client.post("/api/v1/system/reset-db")
assert r.status_code == 200

# Simulate custom fence update endpoint
from backend.api import api_router
from ai.pipeline import CameraPipeline
from video.stream_manager import stream_manager

stream_manager.add_stream("CAM-PERIMETER-TEST", "uploads/20260830_160701_Automatic_Number_Plate_Recognition_(ANPR)___Vehicle_Number_Plate_Recognition_(1).mp4", "MP4")
p = CameraPipeline("CAM-PERIMETER-TEST")
api_router.active_pipelines["CAM-PERIMETER-TEST"] = p

# Send custom fence coordinates (normalized)
fence_payload = {
    "name": "Custom Perimeter Zone",
    "coords": [[0.1, 0.4], [0.9, 0.4], [0.9, 0.9], [0.1, 0.9]],
    "normalized": True
}
update_res = client.post("/api/v1/cameras/CAM-PERIMETER-TEST/fence/update", json=fence_payload)
assert update_res.status_code == 200
data = update_res.json()
assert data["status"] == "success"
assert p.virtual_fence_has_custom is True
print(f"✓ Custom fence armed via API: {data['message']}")

p.start()
time.sleep(5)
p.stop()
stream_manager.remove_stream("CAM-PERIMETER-TEST")
del api_router.active_pipelines["CAM-PERIMETER-TEST"]

# Allow db_worker to commit queued events to sqlite
time.sleep(1.0)
db_worker.stop()

# Fetch alerts as the Perimeter Breach Alerts UI does
alerts_res = client.get("/api/v1/alerts?limit=50")
assert alerts_res.status_code == 200
alerts = alerts_res.json()
print(f"✓ Retrieved {len(alerts)} alerts from /api/v1/alerts")
assert len(alerts) > 0, "Expected alerts to be generated in perimeter breach panel!"

for a in alerts[:5]:
    print(f"  -> [{a.get('severity')}] {a.get('title')}: {a.get('message')}")

print("\n>>> ALL PERIMETER BREACH TESTS PASSED PERFECTLY! <<<")
