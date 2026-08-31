import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8s.pt')
video_path = 'uploads/20260829_135950_car-detection.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    results = model(frame, imgsz=1280, conf=0.01) # Very low confidence to see what it predicts
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            print(f"Detected: {name} (ID: {cls_id}) with confidence: {conf:.4f}")
cap.release()
