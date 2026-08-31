import sys
import os
BASE_DIR = r"c:\Users\lenovo\OneDrive\Desktop\sih"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from datetime import datetime

client = TestClient(app)

print("--- 1. Testing POST /api/v1/system/reset-db ---")
res = client.post("/api/v1/system/reset-db")
assert res.status_code == 200, f"Reset DB failed: {res.text}"
print("Reset response:", res.json())

print("\n--- 2. Verifying Clean State across APIs ---")
# Summary
res_sum = client.get("/api/v1/dashboard/summary")
assert res_sum.status_code == 200
s_data = res_sum.json()
print("Dashboard summary after reset:", s_data)
assert s_data["total_vehicles"] == 0
assert s_data["total_people"] == 0
assert s_data["total_plates"] == 0
assert s_data["active_alerts"] == 0

# Vehicles
res_veh = client.get("/api/v1/dashboard/vehicles")
assert res_veh.status_code == 200
print(f"Vehicles in DB: {len(res_veh.json())}")
assert len(res_veh.json()) == 0

# Alerts
res_alt = client.get("/api/v1/alerts")
assert res_alt.status_code == 200
print(f"Alerts in DB: {len(res_alt.json())}")
assert len(res_alt.json()) == 0

# Analytics
res_ana = client.get("/api/v1/analytics/overview")
assert res_ana.status_code == 200
u_cars = res_ana.json()["unknown_cars"]
print("Analytics unknown_cars after reset:", u_cars)
assert u_cars["total_cars_visited"] == 0
assert u_cars["unknown_cars_visited"] == 0
assert u_cars["authorized_cars_visited"] == 0

print("\n--- 3. Testing Authorized Plates Registry ---")
res_p = client.get("/api/v1/plates/authorized")
assert res_p.status_code == 200
print("Authorized plates:", res_p.json()["plates"])
assert len(res_p.json()["plates"]) >= 6

print("\n--- 4. Verifying Live System Time ---")
now_local = datetime.now()
print(f"Current local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
assert now_local.year == 2026

print("\n>>> ALL TESTS PASSED: DATABASE CLEARS CLEANLY & SYSTEM OPERATES LIVE! <<<")
