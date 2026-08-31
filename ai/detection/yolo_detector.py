from typing import List, Dict, Any
from loguru import logger
import os

try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except Exception as e:
    YOLO = None
    torch = None
    HAS_YOLO = False
    logger.warning(f"YOLO / PyTorch not available in this environment: {e}")

class YOLODetector:
    def __init__(self, model_path: str = None, conf_thresh: float = 0.25):
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.conf_thresh = conf_thresh
        self.allowed_classes = [0, 1, 2, 3, 5, 7]
        self.class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        if HAS_YOLO and YOLO is not None:
            if model_path is None:
                if os.path.exists("models/best.pt"):
                    model_path = "models/best.pt"
                elif os.path.exists("yolov8s.pt"):
                    model_path = "yolov8s.pt"
                elif os.path.exists("yolov8n.pt"):
                    model_path = "yolov8n.pt"
                else:
                    model_path = "yolov8n.pt"
            try:
                logger.info(f"Loading YOLO model '{model_path}' on {self.device}...")
                self.model = YOLO(model_path)
                self.class_names = self.model.names
            except Exception as ex:
                logger.warning(f"Failed to load YOLO model: {ex}")
        
        # Track class-specific confidence thresholds
        # Note: Motorcycle conf is 0.45 to prevent car wheels/hoods from producing noisy motorcycle detections
        self.class_conf = {
            0: 0.22,   # person (0.22 ensures all real people are framed)
            1: 0.22,   # bicycle
            2: 0.20,   # car
            3: 0.30,   # motorcycle
            5: 0.20,   # bus
            7: 0.20,   # truck
        }

    def detect_and_track(self, frame) -> List[Dict[str, Any]]:
        """
        Runs YOLO detection and tracking on a single frame with agnostic NMS and geometric rectification.
        Returns a list of detections with tracking IDs.
        """
        if not self.model:
            return []
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            stream=False,
            verbose=False,
            device=self.device,
            conf=0.18,             # Filter weak background noise
            iou=0.45,              # IoU threshold for NMS
            agnostic_nms=True,     # Suppress overlapping boxes regardless of class
            imgsz=640,             # Standard resolution for fast inference
        )
        raw_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
                
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id not in self.allowed_classes:
                    continue
                
                # Per-class confidence filtering
                min_conf = self.class_conf.get(cls_id, self.conf_thresh)
                if conf < min_conf:
                    continue
                    
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                # Geometric Rectification:
                bw = x2 - x1
                bh = y2 - y1
                aspect = bw / max(bh, 1.0)
                area = bw * bh

                if cls_id == 0:  # person
                    # Filter ground texture artifacts: require minimum height bh >= 14 and aspect ratio bh >= bw * 0.55
                    if (bh < bw * 0.55) or bh < 14:
                        continue
                elif cls_id in [1, 3]:  # bicycle or motorcycle
                    if bw > 175 or area > 32000 or aspect > 1.15:
                        cls_id = 2
                        cls_name = "car"
                elif cls_id == 7:  # truck
                    if area < 55000 and bw < 320 and bh < 230 and aspect < 2.0:
                        cls_id = 2
                        cls_name = "car"
                
                # Determine if this is a vehicle type
                is_vehicle = cls_id in [1, 2, 3, 5, 7]
                
                raw_detections.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "is_vehicle": is_vehicle,
                })
        
        # Inter-class overlap resolution: If a motorcycle detection is largely contained inside a car detection,
        # suppress the motorcycle to prevent duplicate/ghost bike detections on top of cars.
        car_boxes = [d["bbox"] for d in raw_detections if d["class_name"] == "car"]
        final_detections = []
        for d in raw_detections:
            if d["class_name"] in ["motorcycle", "bicycle"]:
                mx1, my1, mx2, my2 = d["bbox"]
                m_area = max(1.0, (mx2 - mx1) * (my2 - my1))
                contained_in_car = False
                for cx1, cy1, cx2, cy2 in car_boxes:
                    ix1 = max(mx1, cx1)
                    iy1 = max(my1, cy1)
                    ix2 = min(mx2, cx2)
                    iy2 = min(my2, cy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if (inter_area / m_area) > 0.40:
                            contained_in_car = True
                            break
                if contained_in_car:
                    continue  # Suppress ghost motorcycle detection on car
            final_detections.append(d)
                
        return final_detections
