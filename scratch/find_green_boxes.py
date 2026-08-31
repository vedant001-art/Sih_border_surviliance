import sys
sys.path.insert(0, ".")
import cv2
from ai.detection.yolo_detector import YOLODetector

detector = YOLODetector()
cap = cv2.VideoCapture("uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4")

for f_idx in range(400):
    ret, frame = cap.read()
    if not ret: break
    dets = detector.detect_and_track(frame)
    for d in dets:
        if d["class_name"] == "person":
            print(f"Frame {f_idx}: Track #{d['track_id']} PERSON conf={d['confidence']:.2f}, bbox={[round(x,1) for x in d['bbox']]}")

cap.release()
