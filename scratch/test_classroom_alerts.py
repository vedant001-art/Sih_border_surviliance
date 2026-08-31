import sys
sys.path.insert(0, ".")
import cv2
import time
from ai.pipeline import CameraPipeline
from video.stream_manager import stream_manager
from backend.services.db_worker import db_worker

db_worker.start()
stream_manager.add_stream("TEST-CLASSROOM", "uploads/20260829_131912_classroom.mp4", "MP4")
p = CameraPipeline("TEST-CLASSROOM")
p.start()

time.sleep(6)

from backend.core.database import SessionLocal
from backend.models.schema import Alert, Event
db = SessionLocal()
alerts = db.query(Alert).all()
print(f"Total Alerts generated on classroom.mp4: {len(alerts)}")
for a in alerts:
    print(f"  Alert: [{a.severity}] {a.title} - {a.message}")
db.close()

p.stop()
stream_manager.remove_stream("TEST-CLASSROOM")
db_worker.stop()
