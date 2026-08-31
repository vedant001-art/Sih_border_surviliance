import cv2
cap = cv2.VideoCapture("uploads/20260830_160701_Automatic_Number_Plate_Recognition_(ANPR)___Vehicle_Number_Plate_Recognition_(1).mp4")
ret, frame = cap.read()
cap.release()
if ret:
    cv2.imwrite("scratch/anpr_sample_frame.jpg", frame)
    print("Saved scratch/anpr_sample_frame.jpg, shape:", frame.shape)
