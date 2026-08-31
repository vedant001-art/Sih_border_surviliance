import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("uploads/20260829_131912_classroom.mp4")
model = YOLO("yolov8s.pt")

ret, frame = cap.read()
cap.release()

res = model.predict(frame, conf=0.45, verbose=False)[0]
people = []
for box in res.boxes:
    cls_id = int(box.cls[0])
    if cls_id == 0:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = x2 - x1, y2 - y1
        ratio = bh / max(bw, 1)
        if ratio >= 0.95 and bh >= 28:
            people.append((box.conf[0].item(), bw, bh, ratio))

print(f"Real people detected in classroom: {len(people)}")
for p in people:
    print(f"  Person: conf={p[0]:.2f}, w={p[1]:.1f}, h={p[2]:.1f}, ratio={p[3]:.2f}")
