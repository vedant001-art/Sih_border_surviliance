import os
import sys
import cv2
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai.detection.yolo_detector import YOLODetector
from ai.detection.plate_detector import PlateDetector
from ai.anpr.plate_reader import ANPRSystem
from backend.services.anpr_fusion import TemporalANPRFusion

video_path = os.path.join(BASE_DIR, "uploads", "20260830_160701_Automatic_Number_Plate_Recognition_(ANPR)___Vehicle_Number_Plate_Recognition_(1).mp4")
if not os.path.exists(video_path):
    print("Video file not found!")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
detector = YOLODetector()
plate_detector = PlateDetector()
anpr_system = ANPRSystem()
fusion = TemporalANPRFusion()

frame_idx = 0
vehicle_tracks = set()
recognized_plates = {}

t0 = time.time()
while cap.isOpened() and frame_idx < 40:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 5 != 0:
        continue
        
    dets = detector.detect_and_track(frame)
    for d in dets:
        if d.get("is_vehicle"):
            tid = d["track_id"]
            vehicle_tracks.add(tid)
            x1, y1, x2, y2 = map(int, d["bbox"])
            h, w = frame.shape[:2]
            v_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if v_crop.size == 0:
                continue
                
            p_det = plate_detector.detect_in_crop(v_crop, max(0, x1), max(0, y1))
            crop = p_det["crop"] if p_det else v_crop[int(v_crop.shape[0]*0.4):, :]
            
            if crop is not None and crop.size > 0:
                res = anpr_system.read_plate(crop)
                if res:
                    fused = fusion.add_observation(tid, res, crop)
                    if fused and fused.get("normalized_text"):
                        recognized_plates[tid] = fused["normalized_text"]

cap.release()
t1 = time.time()

print("="*60, flush=True)
print(f"Benchmark run complete in {t1-t0:.2f}s across {frame_idx} frames.", flush=True)
print(f"Unique vehicles tracked: {len(vehicle_tracks)}", flush=True)
print(f"Vehicles with recognized plates ({len(recognized_plates)}):", flush=True)
for tid, plate in recognized_plates.items():
    print(f"  Track #{tid}: {plate}", flush=True)
print("="*60, flush=True)
