import sys, os
sys.path.insert(0, ".")
import cv2
from ai.detection.yolo_detector import YOLODetector
from ai.detection.plate_detector import PlateDetector
from ai.anpr.plate_reader import ANPRSystem
from backend.services.anpr_fusion import anpr_fusion

video_path = os.path.abspath("uploads/example_vid.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open example_vid.mp4")
    exit(1)

yolo = YOLODetector()
plate_det = PlateDetector()
anpr = ANPRSystem()

frame_idx = 0
found_plates = 0

while cap.isOpened() and frame_idx < 300:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    
    detections = yolo.detect_and_track(frame)
    for d in detections:
        if d.get("is_vehicle"):
            tid = d["track_id"]
            x1, y1, x2, y2 = map(int, d["bbox"])
            h, w = frame.shape[:2]
            veh_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            p_det = plate_det.detect_in_crop(veh_crop, max(0, x1), max(0, y1))
            if p_det:
                crop = p_det["crop"]
            else:
                vh, vw = veh_crop.shape[:2]
                crop = veh_crop[int(vh*0.4):, :] if vh >= 20 and vw >= 20 else veh_crop
                
            res = anpr.read_plate(crop)
            if res:
                fused = anpr_fusion.add_observation(tid, res, crop)
                print(f"Frame {frame_idx} | Track #{tid} | Raw OCR: '{res['raw_text']}' (conf: {res['confidence']:.2f}) -> Normalized: '{res['normalized_text']}' | Fused: '{fused.get('normalized_text') if fused else None}'")
                if fused and fused.get("normalized_text"):
                    found_plates += 1

cap.release()
print(f"Total ANPR plates recognized across 300 frames: {found_plates}")
