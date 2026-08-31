import cv2
import os
import sys
sys.path.insert(0, ".")

from ai.detection.yolo_detector import YOLODetector
from ai.detection.plate_detector import PlateDetector
from ai.anpr.plate_reader import ANPRSystem

cap = cv2.VideoCapture("uploads/example_vid.mp4")
det = YOLODetector()
p_det = PlateDetector()
reader = ANPRSystem()

print(f"Video resolution: {cap.get(3)} x {cap.get(4)}")

frame_idx = 0
found_plates = 0
saved_crops = 0
os.makedirs("scratch/crops", exist_ok=True)

while frame_idx < 120:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 6 != 0:
        continue
    
    dets = det.detect_and_track(frame)
    veh_count = sum(1 for d in dets if d['class_name'] in ['car', 'truck', 'bus', 'vehicle'])
    
    for d in dets:
        if d['class_name'] in ['car', 'truck', 'bus', 'vehicle']:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            # Ensure valid bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            veh_crop = frame[y1:y2, x1:x2]
            if veh_crop.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
                continue
                
            p_res = p_det.detect(veh_crop)
            if p_res and p_res.get('crop') is not None:
                txt = reader.read_plate(p_res['crop'])
                print(f"[Frame {frame_idx}] Plate Detected! Conf: {p_res['confidence']:.2f}, Text: '{txt}'")
                if txt:
                    found_plates += 1
                if saved_crops < 5:
                    cv2.imwrite(f"scratch/crops/plate_crop_{frame_idx}_{saved_crops}.jpg", p_res['crop'])
                    saved_crops += 1
            else:
                # Test lower region
                vh, vw = veh_crop.shape[:2]
                lower_crop = veh_crop[int(vh*0.5):, :]
                txt = reader.read_plate(lower_crop)
                if txt:
                    print(f"[Frame {frame_idx}] Lower crop fallback Text: '{txt}'")
                    found_plates += 1

cap.release()
print(f"\nTotal frames checked: {frame_idx}, Total plates read: {found_plates}")
