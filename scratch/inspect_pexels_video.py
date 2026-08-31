import sys
sys.path.insert(0, ".")
import cv2
from ultralytics import YOLO

video_path = r"uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8s.pt")

print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

person_detections = []
for f_idx in range(0, 300, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break
    
    res = model.predict(frame, conf=0.25, verbose=False)[0]
    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0: # person
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            person_detections.append((f_idx, conf, bw, bh, box.xyxy[0].tolist()))

print(f"Total person detections found across sampled frames: {len(person_detections)}")
for d in person_detections[:10]:
    print(f"Frame {d[0]}: conf={d[1]:.2f}, w={d[2]:.1f}, h={d[3]:.1f}, ratio={d[3]/max(d[2],1):.2f}, bbox={[round(x,1) for x in d[4]]}")

cap.release()
