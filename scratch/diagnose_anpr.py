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
    print(f"Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Loaded video: {video_path}", flush=True)
print(f"Resolution: {int(cap.get(3))}x{int(cap.get(4))}, Total frames: {total_frames}, FPS: {fps}", flush=True)

detector = YOLODetector()
plate_detector = PlateDetector()
anpr_system = ANPRSystem()
fusion = TemporalANPRFusion()

frame_idx = 0
vehicle_tracks = set()
plates_detected = 0
plates_ocr_success = 0
fused_results = {}

while cap.isOpened() and frame_idx < 60:
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
            if p_det and p_det.get("crop") is not None:
                plates_detected += 1
                p_crop = p_det["crop"]
                p_conf = p_det["confidence"]
                p_res = anpr_system.read_plate(p_crop)
                if p_res:
                    plates_ocr_success += 1
                    fused = fusion.add_observation(tid, p_res, p_crop)
                    if fused:
                        fused_results[tid] = fused
                    print(f"Frame {frame_idx:03d} | Track #{tid:02d} | Plate YOLO conf: {p_conf:.2f} | Raw: '{p_res['raw_text']}' | Norm: '{p_res['normalized_text']}' (OCR conf: {p_res['confidence']:.2f})", flush=True)
            else:
                vh, vw = v_crop.shape[:2]
                if vh > 30 and vw > 30:
                    lower_crop = v_crop[int(vh * 0.4):, :]
                    p_res = anpr_system.read_plate(lower_crop)
                    if p_res:
                        plates_ocr_success += 1
                        fused = fusion.add_observation(tid, p_res, lower_crop)
                        if fused:
                            fused_results[tid] = fused
                        print(f"Frame {frame_idx:03d} | Track #{tid:02d} | [Fallback Lower] | Raw: '{p_res['raw_text']}' | Norm: '{p_res['normalized_text']}' (OCR conf: {p_res['confidence']:.2f})", flush=True)

cap.release()

print("\n" + "="*60)
print(f"Total Unique Vehicle Tracks: {len(vehicle_tracks)}")
print(f"Plate Detection Hits: {plates_detected}")
print(f"Plate OCR Successes: {plates_ocr_success}")
print(f"Final Fused Vehicle Plates:")
for tid, fused in fused_results.items():
    print(f"  Track #{tid}: '{fused.get('normalized_text')}' (Raw: '{fused.get('raw_text')}', conf: {fused.get('confidence'):.2f}, valid_ind: {fused.get('is_valid')})")
print("="*60)
