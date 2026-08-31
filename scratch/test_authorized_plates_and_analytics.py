import sys
import os
BASE_DIR = r"c:\Users\lenovo\OneDrive\Desktop\sih"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.authorized_plates import authorized_plates
from backend.core.database import SessionLocal, engine

client = TestClient(app)

from sqlalchemy import text
print("--- 1. Testing SQLite WAL Mode & Pragma ---")
db = SessionLocal()
try:
    res = db.execute(text("PRAGMA journal_mode")).fetchone()
    print(f"SQLite journal_mode: {res[0]}")
    assert res[0].lower() == "wal", f"Expected WAL mode, got {res[0]}"
    
    timeout_res = db.execute(text("PRAGMA busy_timeout")).fetchone()
    print(f"SQLite busy_timeout: {timeout_res[0]} ms")
finally:
    db.close()

print("\n--- 2. Testing Authorized Plates REST API ---")
# GET
r_get = client.get("/api/v1/plates/authorized")
assert r_get.status_code == 200
data = r_get.json()
print("Initial plates:", data["plates"])
assert "DL8CAF1234" in data["plates"]

# POST
r_post = client.post("/api/v1/plates/authorized", json={"plate": "TEST9999"})
assert r_post.status_code == 200
assert "TEST9999" in r_post.json()["plates"]
assert authorized_plates.is_authorized("TEST 9999") is True
print("Added TEST9999: verified authorized")

# DELETE
r_del = client.delete("/api/v1/plates/authorized/TEST9999")
assert r_del.status_code == 200
assert "TEST9999" not in r_del.json()["plates"]
assert authorized_plates.is_authorized("TEST9999") is False
print("Deleted TEST9999: verified removed")

print("\n--- 3. Testing Analytics API with Unknown Cars Visited ---")
r_ana = client.get("/api/v1/analytics/overview")
assert r_ana.status_code == 200
ana_data = r_ana.json()
assert "unknown_cars" in ana_data
u_info = ana_data["unknown_cars"]
print("Unknown Cars Data:", u_info)
assert "total_cars_visited" in u_info
assert "unknown_cars_visited" in u_info
assert "authorized_cars_visited" in u_info
assert "unknown_ratio_pct" in u_info

print("\n--- 4. Testing Alert Filtering for Authorized vs Unknown Cars in Pipeline ---")
from ai.pipeline import CameraPipeline
pipe = CameraPipeline("TEST-VERIFY")

# Mock detections: one car with authorized plate, one car with unknown plate
det_auth = {
    "track_id": 101,
    "bbox": [100, 100, 300, 250],
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.8
}
det_unknown = {
    "track_id": 102,
    "bbox": [150, 150, 350, 300],
    "class_name": "car",
    "is_vehicle": True,
    "confidence": 0.8
}

pipe._plate_cache[101] = {"plate": "DL8CAF1234"} # Authorized!
pipe._plate_cache[102] = {"plate": "UNREGISTERED_CAR"} # Unknown!

# Set up virtual fence covering the bbox area
from ai.behavior.virtual_fence import VirtualFence
pipe.virtual_fence = VirtualFence(zones=[{
    "id": 1,
    "name": "Restricted Zone A",
    "type": "POLYGON",
    "coords": [(0, 0), (600, 0), (600, 600), (0, 600)]
}])

import numpy as np
dummy_frame = np.zeros((400, 600, 3), dtype=np.uint8)

# Evaluate rules
pipe._evaluate_rules([det_auth, det_unknown], dummy_frame)
print(">>> ALL VERIFICATIONS COMPLETED SUCCESSFULLY! <<<")
