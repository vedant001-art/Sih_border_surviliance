import sys
sys.path.insert(0, ".")
import time
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.offline_alert_queue import offline_alert_queue
from backend.services.db_worker import db_worker

db_worker.start()
client = TestClient(app)

print("=== 1. Testing Load Example Video (pexels-casey-whalen-6571483) ===")
res = client.post("/api/v1/cameras/load-example")
assert res.status_code == 200
data = res.json()
print("Response:", data)
assert data["status"] == "success"
assert data["name"] == "Example Vid"
assert data["camera_id"] == "CAM-01"

# Check camera status
cams_res = client.get("/api/v1/dashboard/cameras-status")
assert cams_res.status_code == 200
cams = cams_res.json()
cam01 = next((c for c in cams if c["camera_id"] == "CAM-01"), None)
assert cam01 is not None
print(f"✓ CAM-01 Status: {cam01['status']}, Name: {cam01['name']}, Location: {cam01['location']}")
assert cam01["name"] == "Example Vid"
assert cam01["status"] == "ONLINE"

print("\n=== 2. Testing Store-and-Forward Offline Alert Queue ===")
# A. Turn Data connection OFF
toggle_res = client.post("/api/v1/alerts/toggle-data-connection", json={"connected": False})
assert toggle_res.status_code == 200
assert offline_alert_queue.is_data_connected is False
print("✓ Data Connection set to OFF (Offline Queue Buffering Mode)")

# B. Let pipeline run for 4 seconds to detect vehicles and produce alerts while data is OFF
time.sleep(4)

stats_res = client.get("/api/v1/alerts/offline-queue-stats")
assert stats_res.status_code == 200
stats = stats_res.json()
print(f"✓ Offline Queue Stats while data is OFF: {stats}")
assert stats["is_data_connected"] is False
assert stats["queue_length"] > 0, f"Expected alerts in queue, got {stats['queue_length']}"
print(f"✓ Successfully buffered {stats['queue_length']} alerts in local FIFO queue while offline!")

# C. Reconnect and sync all alerts from queue
print("\n=== 3. Testing Reconnection & Queue Synchronization ===")
sync_res = client.post("/api/v1/alerts/sync-offline-queue")
assert sync_res.status_code == 200
sync_data = sync_res.json()
print(f"✓ Reconnected! Synced {sync_data['synced_count']} alerts from queue:")
for a in sync_data["alerts"][:3]:
    print(f"  -> Alert #{a.get('id')}: {a.get('title')} (Buffered: {a.get('buffered_offline')})")

assert offline_alert_queue.count() == 0, "Queue should be completely flushed after sync"
assert offline_alert_queue.is_data_connected is True, "Data connection should be restored"
print("✓ Queue drained to 0 and data connection restored to ONLINE.")

# Stop camera and worker
client.post("/api/v1/cameras/CAM-01/stop")
db_worker.stop()
print("\n>>> ALL EXAMPLE VIDEO & OFFLINE QUEUE TESTS PASSED PERFECTLY! <<<")
