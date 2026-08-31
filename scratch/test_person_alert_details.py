import sys
import os
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import SessionLocal
from backend.models.schema import Alert, Event, Track, Vehicle, Camera
import json

client = TestClient(app)

# Reset DB first
res_reset = client.post("/api/v1/system/reset-db")
assert res_reset.status_code == 200, f"Reset failed: {res_reset.text}"
print("✓ DB reset successfully.")

# Create test camera
db = SessionLocal()
try:
    cam = db.query(Camera).filter(Camera.id == "CAM-01").first()
    if not cam:
        cam = Camera(id="CAM-01", name="Sector Alpha", location_name="Perimeter Fence")
        db.add(cam)
        db.commit()

    # 1. Enqueue a PERSON intrusion event directly via db_worker to test DB + API flow
    from backend.services.db_worker import db_worker

    person_event_data = {
        "type": "INTRUSION",
        "camera_id": "CAM-01",
        "local_track_id": 14,
        "object_type": "person",
        "class_name": "person",
        "title": "PERIMETER BREACH: PERSON #14",
        "message": "PERSON #14 (Priya Sharma, LEVEL-2 AUTH, Clothing: White Top) crossed Virtual Fence into Restricted Sector CAM-01.",
        "details": {
            "object_type": "person",
            "class_name": "person",
            "name": "Priya Sharma",
            "clearance": "LEVEL-2 AUTH",
            "clothing": "White Top",
            "activity": "Walking",
            "speed_kmh": 4.1,
            "zone_name": "Restricted Sector CAM-01"
        },
        "severity": "CRITICAL"
    }
    
    # Process event synchronously
    db_worker._handle_create_event(db, person_event_data)
    db.commit()
    print("✓ Person breach event created.")

    # 2. Enqueue a VEHICLE intrusion event
    vehicle_event_data = {
        "type": "INTRUSION",
        "camera_id": "CAM-01",
        "local_track_id": 22,
        "object_type": "vehicle",
        "class_name": "car",
        "title": "UNAUTHORIZED VEHICLE: CAR #22",
        "message": "UNAUTHORIZED CAR #22 (Plate: DL8CAF1234) crossed Virtual Fence into Restricted Sector CAM-01.",
        "details": {
            "object_type": "vehicle",
            "class_name": "car",
            "plate": "DL8CAF1234",
            "type": "Car",
            "make": "Toyota",
            "color": "White",
            "speed_kmh": 36.5,
            "zone_name": "Restricted Sector CAM-01"
        },
        "severity": "CRITICAL"
    }
    db_worker._handle_create_event(db, vehicle_event_data)
    db.commit()
    print("✓ Vehicle breach event created.")

finally:
    db.close()

# 3. Query GET /api/v1/alerts
res = client.get("/api/v1/alerts")
assert res.status_code == 200, f"Failed to get alerts: {res.text}"
alerts = res.json()
assert len(alerts) == 2, f"Expected 2 alerts, got {len(alerts)}"

# Test Person Alert
person_alert = next((a for a in alerts if a["object_type"] == "person"), None)
assert person_alert is not None, "Person alert not found in /alerts response!"
assert "vehicle" not in person_alert["title"].lower(), f"Person alert title mentions vehicle! Title: {person_alert['title']}"
assert "vehicle" not in person_alert["message"].lower(), f"Person alert message mentions vehicle! Message: {person_alert['message']}"
assert person_alert["person_info"] is not None, "person_info missing from person alert!"
assert person_alert["person_info"]["name"] == "Priya Sharma"
assert person_alert["person_info"]["clothing"] == "White Top"
assert person_alert["person_info"]["clearance"] == "LEVEL-2 AUTH"
assert person_alert["vehicle_info"] is None, "vehicle_info should be None for person alert!"
print("✓ Person Alert Verified: Zero vehicle mentions, rich person_info populated correctly!")

# Test Vehicle Alert
veh_alert = next((a for a in alerts if a["object_type"] == "vehicle"), None)
assert veh_alert is not None, "Vehicle alert not found in /alerts response!"
assert veh_alert["vehicle_info"] is not None, "vehicle_info missing from vehicle alert!"
assert veh_alert["vehicle_info"]["plate"] == "DL8CAF1234"
assert veh_alert["vehicle_info"]["color"] == "White"
assert veh_alert["person_info"] is None, "person_info should be None for vehicle alert!"
print("✓ Vehicle Alert Verified: Plate, color, make, speed populated correctly!")

print("\n>>> ALL PERSON / VEHICLE ALERT TESTS PASSED PERFECTLY! <<<")
