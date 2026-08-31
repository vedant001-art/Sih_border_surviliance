import sys
sys.path.insert(0, ".")
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4")
model = YOLO("yolov8s.pt")

phantom_count = 0
total_checked = 0

for f_idx in range(0, 300, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break
    
    total_checked += 1
    res = model.predict(frame, conf=0.45, verbose=False)[0]
    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0: # person
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            ratio = bh / max(bw, 1)
            # Apply aspect ratio check: real upright humans have bh >= bw * 0.95
            if ratio < 0.95 or bh < 28:
                continue
            phantom_count += 1
            print(f"Frame {f_idx}: PERSON conf={conf:.2f}, w={bw:.1f}, h={bh:.1f}, ratio={ratio:.2f}")

cap.release()
print(f"\nTested {total_checked} sampled frames from 4K highway video.")
print(f"Phantom persons detected: {phantom_count}")
