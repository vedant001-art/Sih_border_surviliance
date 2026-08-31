import hashlib
import cv2
import numpy as np

class VehicleAttributeEngine:
    def __init__(self):
        self.types = ["Sedan", "SUV", "Hatchback", "Commercial Truck", "Motorcycle"]
        self.makes = ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota"]
        
    def _detect_color(self, crop: np.ndarray) -> str:
        if crop is None or crop.size == 0: return "Unknown"
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Use center region to avoid background
        h, w = hsv.shape[:2]
        center = hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        if center.size == 0: center = hsv
        
        avg_hue = np.median(center[:,:,0])
        avg_sat = np.median(center[:,:,1])
        avg_val = np.median(center[:,:,2])
        
        if avg_val < 50: return "Black"
        if avg_val > 200 and avg_sat < 40: return "White"
        if avg_sat < 50: return "Silver/Grey"
        
        if avg_hue < 10 or avg_hue > 160: return "Red"
        if 10 <= avg_hue < 35: return "Orange/Yellow"
        if 35 <= avg_hue < 85: return "Green"
        if 85 <= avg_hue < 140: return "Blue"
        return "Purple/Pink"

    def get_vehicle_profile(self, track_id: int, class_name: str, crop: np.ndarray = None) -> dict:
        """Deterministically hashes the track_id to assign stable vehicle attributes, but detects actual color"""
        hash_val = hashlib.md5(str(track_id).encode()).hexdigest()
        
        color = self._detect_color(crop) if crop is not None else "Unknown"
        
        # Crop dimension check: If crop is wide/large, it's a four-wheeler car, not a motorcycle
        if crop is not None and crop.size > 0:
            ch, cw = crop.shape[:2]
            aspect = cw / max(ch, 1)
            if (cw > 170 or (cw * ch) > 30000 or aspect > 1.15) and class_name in ["motorcycle", "bicycle"]:
                class_name = "car"
        
        if class_name == "car":
            v_type = self.types[int(hash_val[-2], 16) % 3] # Sedan, SUV, Hatchback
            make = self.makes[int(hash_val[-3], 16) % len(self.makes)]
        elif class_name == "truck":
            v_type = "Commercial Truck"
            make = "Tata" if int(hash_val[-2], 16) % 2 == 0 else "Ashok Leyland"
        elif class_name == "motorcycle" or class_name == "bicycle":
            v_type = "Motorcycle" if class_name == "motorcycle" else "Bicycle"
            make = "Bajaj" if class_name == "motorcycle" else "Hero"
        else:
            v_type = "Heavy Vehicle"
            make = "Unknown"
            
        return {
            "type": v_type,
            "color": color,
            "make": make
        }

vehicle_attribute_engine = VehicleAttributeEngine()
