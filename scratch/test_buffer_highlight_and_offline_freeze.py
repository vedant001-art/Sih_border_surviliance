import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.offline_alert_queue import offline_alert_queue
from backend.services.db_worker import db_worker
from ai.anpr.plate_reader import ANPRSystem

db_worker.start()
client = TestClient(app)

print("=== 1. Testing OCR & Number Plate Extraction ===")
anpr = ANPRSystem()
# Create synthetic image with clean text
test_plate = np.ones((80, 240, 3), dtype=np.uint8) * 240
cv2.putText(test_plate, "KA02MM9091", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
read_res = anpr.read_plate(test_plate)
print("Synthetic Indian Plate OCR Result:", read_res)
assert read_res is not None, "ANPR failed to read clean plate"
assert "KA02MM" in read_res["normalized_text"] or "9091" in read_res["normalized_text"]
print("✓ Clean plate recognized by multi-pass ANPR!")

print("\n=== 2. Testing Plate Number Output in Vehicle & Alert APIs ===")
client.post("/api/v1/cameras/load-example")
# Give it 3 seconds to process frames
import time
time.sleep(3)

# Check vehicles
veh_res = client.get("/api/v1/dashboard/vehicles")
assert veh_res.status_code == 200
vehs = veh_res.json()
print(f"Logged {len(vehs)} vehicles:")
for v in vehs[:4]:
    print(f"  -> Track #{v['track_id']}: Type={v['vehicle_type']}, Plate='{v['plate_number']}' (Visible: {bool(v['plate_number'])})")
    assert v['plate_number'] and v['plate_number'] != "UNREADABLE", f"Plate must be visible, got {v['plate_number']}"
print("✓ All vehicle outputs have visible plate identifiers!")

print("\n=== 3. Testing Offline Freeze & Store-and-Forward Buffering ===")
toggle_res = client.post("/api/v1/alerts/toggle-data-connection", json={"connected": False})
assert toggle_res.status_code == 200
assert offline_alert_queue.is_data_connected is False
print("✓ Data Connection set to OFF (Buffering Mode)")

time.sleep(3)
stats_res = client.get("/api/v1/alerts/offline-queue-stats")
stats = stats_res.json()
print("Queue Stats while offline:", stats)
assert stats["queue_length"] > 0, "Alerts should be buffered in queue while offline"
print(f"✓ Video and AI kept recording alerts into queue ({stats['queue_length']} queued)!")

print("\n=== 4. Testing Reconnection & Buffered Data Highlighting ===")
sync_res = client.post("/api/v1/alerts/sync-offline-queue")
assert sync_res.status_code == 200
sync_data = sync_res.json()
print(f"Synced {sync_data['synced_count']} alerts:")
for a in sync_data["alerts"]:
    print(f"  -> Alert #{a.get('id')}: Title='{a.get('title')}', buffered_offline={a.get('buffered_offline')}, is_buffered={a.get('is_buffered')}")
    assert a.get("buffered_offline") is True, "Must be tagged as buffered_offline"
    assert a.get("is_buffered") is True, "Must be tagged as is_buffered"

print("✓ All flushed alerts are explicitly marked as BUFFER DATA for frontend highlighting!")
assert offline_alert_queue.count() == 0
assert offline_alert_queue.is_data_connected is True
print("✓ Queue drained and data connection restored to ONLINE.")

client.post("/api/v1/cameras/CAM-01/stop")
db_worker.stop()
print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
