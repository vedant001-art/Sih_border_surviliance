import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
video_path = 'uploads/20260829_135950_car-detection.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2) # Go to middle of video
ret, frame = cap.read()
if ret:
    results = model(frame, imgsz=1280, conf=0.01)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            print(f"Detected: {name} (ID: {cls_id}) with confidence: {conf:.4f}")
cap.release()
