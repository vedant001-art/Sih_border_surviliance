import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
video_path = 'uploads/20260829_135950_car-detection.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
ret, frame = cap.read()
if ret:
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imwrite('uploads/debug_frame.jpg', annotated_frame)
cap.release()
