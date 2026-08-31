import cv2
import os

cap = cv2.VideoCapture("uploads/20260830_211902_pexels-casey-whalen-6571483_(2160p).mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if ret:
    x1, y1, x2, y2 = 1230, 1106, 1402, 1224
    crop = frame[y1:y2, x1:x2]
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite("scratch/phantom_road_crop.jpg", crop)
    print(f"Saved phantom road crop: shape={crop.shape}, mean color={crop.mean(axis=(0,1))}")
