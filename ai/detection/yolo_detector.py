try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except Exception:
    YOLO = None
    torch = None
    HAS_YOLO = False

import math
from typing import List, Dict, Any
from loguru import logger
import os

class YOLODetector:
    def __init__(self, model_path: str = None, conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.allowed_classes = [0, 1, 2, 3, 5, 7]
        self.class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.class_conf = {0: 0.40, 1: 0.30, 2: 0.18, 3: 0.35, 5: 0.30, 7: 0.30}
        self._last_centers = {}
        self._next_fallback_id = 1000

        if not HAS_YOLO:
            self.device = "cpu"
            self.model = None
            logger.info("Running in lightweight CPU fallback mode for YOLODetector.")
            return

        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        
        # Auto-select model
        if model_path is None:
            if os.path.exists("models/best.pt"):
                model_path = "models/best.pt"
            elif os.path.exists("yolov8s.pt"):
                model_path = "yolov8s.pt"
            elif os.path.exists("yolov8n.pt"):
                model_path = "yolov8n.pt"
            else:
                model_path = "yolov8n.pt"
        
        self.model_path = model_path
        try:
            logger.info(f"Initializing dedicated YOLO tracker '{model_path}' on {self.device}...")
            self.model = YOLO(model_path)
            self.class_names = self.model.names
        except Exception as e:
            logger.warning(f"Failed to load YOLO model ({e}). Using lightweight fallback detector.")
            self.model = None

        
        # Track class-specific confidence thresholds
        self.class_conf = {
            0: 0.40,   # person
            1: 0.30,   # bicycle
            2: 0.18,   # car
            3: 0.35,   # motorcycle
            5: 0.25,   # bus
            7: 0.25,   # truck
        }
        
        self._fallback_id_counter = 1
        self._last_centers: Dict[int, tuple] = {}
        self._smoothed_boxes: Dict[int, tuple] = {}

    def _assign_fallback_track_id(self, cx: float, cy: float) -> int:
        best_id = None
        min_dist = 100.0  # max 100px movement threshold
        for tid, (last_x, last_y) in self._last_centers.items():
            dist = math.hypot(cx - last_x, cy - last_y)
            if dist < min_dist:
                min_dist = dist
                best_id = tid
        
        if best_id is None:
            best_id = self._fallback_id_counter
            self._fallback_id_counter += 1
            
        self._last_centers[best_id] = (cx, cy)
        return best_id

    def detect_and_track(self, frame) -> List[Dict[str, Any]]:
        """
        Runs YOLO detection and tracking on a single frame.
        """
        if self.model is None or frame is None:
            return []

        results = self.model.track(

            frame,
            persist=True,
            tracker="bytetrack.yaml",
            stream=False,
            verbose=False,
            device=self.device,
            conf=0.15,
            iou=0.45,
            agnostic_nms=True,
            imgsz=640,
        )
        raw_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            has_track_ids = boxes.id is not None
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id not in self.allowed_classes:
                    continue
                
                min_conf = self.class_conf.get(cls_id, self.conf_thresh)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                if has_track_ids and i < len(boxes.id):
                    track_id = int(boxes.id[i])
                    self._last_centers[track_id] = (cx, cy)
                else:
                    track_id = self._assign_fallback_track_id(cx, cy)
                
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                # Geometric Rectification:
                bw = x2 - x1
                bh = y2 - y1
                aspect = bw / max(bh, 1.0)
                area = bw * bh

                if cls_id == 0:  # person
                    if (bh < bw * 0.90) or bh < 24 or bw < 10:
                        continue
                elif cls_id in [1, 3]:  # bicycle or motorcycle
                    if bw > 175 or area > 32000 or aspect > 1.15:
                        cls_id = 2
                        cls_name = "car"
                elif cls_id == 7:  # truck
                    if area < 55000 and bw < 320 and bh < 230 and aspect < 2.0:
                        cls_id = 2
                        cls_name = "car"
                
                is_vehicle = cls_id in [1, 2, 3, 5, 7]
                
                if bw < 8 or bh < 8:
                    continue

                raw_detections.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "is_vehicle": is_vehicle,
                })
        
        # Inter-class overlap resolution
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
                    continue
            final_detections.append(d)

        # EMA Bounding Box Smoothing per Track ID for rock-solid, jitter-free target tracking
        smoothed_detections = []
        alpha = 0.65  # Weight for new frame detection vs historical position
        active_tids = set()
        for d in final_detections:
            tid = d["track_id"]
            active_tids.add(tid)
            nx1, ny1, nx2, ny2 = d["bbox"]
            if tid in self._smoothed_boxes:
                ox1, oy1, ox2, oy2 = self._smoothed_boxes[tid]
                sx1 = alpha * nx1 + (1 - alpha) * ox1
                sy1 = alpha * ny1 + (1 - alpha) * oy1
                sx2 = alpha * nx2 + (1 - alpha) * ox2
                sy2 = alpha * ny2 + (1 - alpha) * oy2
            else:
                sx1, sy1, sx2, sy2 = nx1, ny1, nx2, ny2
            
            self._smoothed_boxes[tid] = (sx1, sy1, sx2, sy2)
            d["bbox"] = [sx1, sy1, sx2, sy2]
            smoothed_detections.append(d)

        # Cleanup lost tracks
        lost_tids = [t for t in self._smoothed_boxes if t not in active_tids]
        for t in lost_tids:
            del self._smoothed_boxes[t]
                
        return smoothed_detections
