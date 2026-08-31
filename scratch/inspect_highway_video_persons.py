import sys
import os
sys.path.insert(0, ".")
import cv2
from ai.detection.yolo_detector import YOLODetector

video_path = r"uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4"
cap = cv2.VideoCapture(video_path)
detector = YOLODetector()

for frame_idx in range(40):
    ret, frame = cap.read()
    if not ret:
        break
    dets = detector.detect_and_track(frame)
    for d in dets:
        if d["class_name"] == "person":
            print(f"Frame {frame_idx}: Track {d['track_id']} PERSON conf={d['confidence']:.2f} bbox={d['bbox']}")

cap.release()
