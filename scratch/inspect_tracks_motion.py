import sys
sys.path.insert(0, ".")
import cv2
from ai.detection.yolo_detector import YOLODetector

detector = YOLODetector()
cap = cv2.VideoCapture("uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4")

print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

tracks_seen = {}
for f_idx in range(120):
    ret, frame = cap.read()
    if not ret: break
    dets = detector.detect_and_track(frame)
    for d in dets:
        tid = d["track_id"]
        cname = d["class_name"]
        bbox = [round(x, 1) for x in d["bbox"]]
        if tid not in tracks_seen:
            tracks_seen[tid] = {"class": cname, "frames": 0, "first_f": f_idx, "last_f": f_idx, "first_bbox": bbox, "last_bbox": bbox}
        tracks_seen[tid]["frames"] += 1
        tracks_seen[tid]["last_f"] = f_idx
        tracks_seen[tid]["last_bbox"] = bbox

cap.release()

print("\n--- Tracks summary in first 120 frames ---")
for tid, info in tracks_seen.items():
    dx = abs(info["last_bbox"][0] - info["first_bbox"][0])
    dy = abs(info["last_bbox"][1] - info["first_bbox"][1])
    is_stationary = (dx < 5 and dy < 5 and info["frames"] > 10)
    print(f"Track #{tid} [{info['class']}]: frames={info['frames']}, span={info['first_f']}->{info['last_f']}, dx={dx:.1f}, dy={dy:.1f}, stationary={is_stationary}, bbox={info['first_bbox']}")
