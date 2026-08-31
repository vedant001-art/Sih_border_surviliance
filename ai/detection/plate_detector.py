from ultralytics import YOLO
import torch
from loguru import logger
import os
import cv2
import numpy as np
import threading
from typing import Optional, Dict, Any

class PlateDetector:
    _shared_models: Dict[str, YOLO] = {}
    _shared_lock = threading.Lock()

    def __init__(self, model_path: str = None, conf_thresh: float = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detect best model available
        if model_path is None:
            if os.path.exists("models/anpr_best.pt"):
                model_path = "models/anpr_best.pt"
                default_conf = 0.25
            elif os.path.exists("models/best.pt"):
                model_path = "models/best.pt"
                default_conf = 0.25
            elif os.path.exists("models/best_epoch22_undertrained.pt"):
                model_path = "models/best_epoch22_undertrained.pt"
                default_conf = 0.10
            else:
                model_path = "models/best_epoch22_undertrained.pt"
                default_conf = 0.10
        else:
            default_conf = 0.10
            
        self.conf_thresh = conf_thresh if conf_thresh is not None else default_conf
        self.model = None
        self.enabled = False
        
        if os.path.exists(model_path):
            try:
                with PlateDetector._shared_lock:
                    if model_path not in PlateDetector._shared_models:
                        logger.info(f"Loading Plate YOLO model '{model_path}' (conf={self.conf_thresh}) on {self.device}...")
                        PlateDetector._shared_models[model_path] = YOLO(model_path)
                    self.model = PlateDetector._shared_models[model_path]
                self.enabled = True
            except Exception as e:
                logger.error(f"Failed to load plate model: {e}")
        else:
            logger.warning(f"License plate model weights '{model_path}' are required for ANPR. Plate detector is disabled.")

    @torch.inference_mode()
    def detect_in_crop(self, vehicle_crop: np.ndarray, global_offset_x: int, global_offset_y: int) -> Optional[Dict[str, Any]]:
        """
        Runs plate detection inside a vehicle crop to localize the plate.
        Returns coordinates mapped back to the global frame, or None if no plate found.
        """
        if not self.enabled or vehicle_crop is None or vehicle_crop.size == 0:
            return None
            
        h, w = vehicle_crop.shape[:2]
        
        # Adaptive scaling: if crop is very small, upscale it for the detector
        scale_factor = 1.0
        if w < 300 or h < 300:
            scale_factor = 2.0
            inference_img = cv2.resize(vehicle_crop, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
        else:
            inference_img = vehicle_crop
            
        results = self.model.predict(
            inference_img,
            device=self.device,
            conf=0.15,
            iou=0.45,
            imgsz=640,
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            return None
            
        # Get the highest confidence plate
        best_box = None
        best_conf = 0.0
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].tolist()
                
        if best_box is None:
            return None
            
        # Map back to original vehicle crop scale
        x1_local, y1_local, x2_local, y2_local = [coord / scale_factor for coord in best_box]
        
        # Map back to global frame coordinates
        gx1 = int(x1_local + global_offset_x)
        gy1 = int(y1_local + global_offset_y)
        gx2 = int(x2_local + global_offset_x)
        gy2 = int(y2_local + global_offset_y)
        
        return {
            "bbox": [gx1, gy1, gx2, gy2],
            "confidence": best_conf,
            "crop": vehicle_crop[int(y1_local):int(y2_local), int(x1_local):int(x2_local)].copy()
        }
