import os
import sys
import time

# Ensure project root is in sys.path for linter and runtime
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
from loguru import logger
from ai.pipeline import CameraPipeline
from backend.services.db_worker import db_worker
from video.stream_manager import stream_manager
from backend.core.database import SessionLocal
from backend.models.schema import Vehicle, ANPRRecord, Event

def test_pipeline():
    logger.info("Starting ANPR test pipeline...")
    video_path = os.path.join(BASE_DIR, "uploads", "20260830_160701_Automatic_Number_Plate_Recognition_(ANPR)___Vehicle_Number_Plate_Recognition_(1).mp4")
    
    if not os.path.exists(video_path):
        import glob
        mp4_files = sorted(glob.glob(os.path.join(BASE_DIR, "uploads", "*.mp4")), key=os.path.getmtime, reverse=True)
        if mp4_files:
            video_path = mp4_files[0]
            logger.info(f"Selected latest uploaded video: {video_path}")
        else:
            logger.error("No MP4 video files found in uploads directory.")
            sys.exit(1)
        
    db_worker.start()
    stream_manager.add_stream("TEST-01", video_path, "MP4")
    
    pipeline = CameraPipeline("TEST-01")
    pipeline.start()
    
    logger.info("Processing frames...")
    # Let it run for 15 seconds to accumulate data
    time.sleep(15)
    
    pipeline.stop()
    stream_manager.remove_stream("TEST-01")
    db_worker.stop()
    
    # Query stats
    db = SessionLocal()
    try:
        v_count = db.query(Vehicle).count()
        a_count = db.query(ANPRRecord).count()
        e_count = db.query(Event).filter(Event.event_type == 'PLATE_DETECTED').count()
        
        avg_plate_conf = 0.0
        avg_ocr_conf = 0.0
        records = db.query(ANPRRecord).all()
        if records:
            avg_plate_conf = sum(getattr(r, 'plate_detection_confidence', 0) or 0 for r in records) / len(records)
            avg_ocr_conf = sum(getattr(r, 'ocr_confidence', 0) or 0 for r in records) / len(records)
    finally:
        db.close()
        
    latency = getattr(pipeline, 'inference_latency_ms', 0)
    
    print("\n" + "="*50)
    print("ANPR DIAGNOSTIC RESULTS")
    print("="*50)
    print(f"Vehicles tracked: {v_count}")
    print(f"ANPR Records created: {a_count}")
    print(f"Plate Detection Events: {e_count}")
    print(f"Avg Plate Confidence: {avg_plate_conf*100:.1f}%")
    print(f"Avg OCR Confidence: {avg_ocr_conf*100:.1f}%")
    print(f"Avg ANPR Latency: {latency} ms")
    print("="*50)

if __name__ == '__main__':
    test_pipeline()

