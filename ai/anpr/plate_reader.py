import os
from loguru import logger
import re
import cv2
import easyocr
import numpy as np
import threading

class ANPRSystem:
    _shared_ocr = None
    _shared_ocr_lock = threading.Lock()

    def __init__(self):
        self.enabled = True
        if ANPRSystem._shared_ocr is None:
            with ANPRSystem._shared_ocr_lock:
                if ANPRSystem._shared_ocr is None:
                    try:
                        logger.info("Loading EasyOCR for ANPR...")
                        ANPRSystem._shared_ocr = easyocr.Reader(['en'], gpu=False)
                        logger.info("EasyOCR loaded for ANPR")
                    except Exception as e:
                        logger.error(f"Failed to load EasyOCR: {e}")
        self.ocr = ANPRSystem._shared_ocr
        if not self.ocr:
            self.enabled = False

    def validate_indian_plate(self, text: str) -> bool:
        """
        Validates if text roughly matches Indian Number Plate formats.
        Examples: UP16BZ0345, MH01AE8017, DL8CAF5030, KL07CD4321
        """
        # Remove spaces and non-alphanumeric chars
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Indian standard formats (MH12AB1234, KA05CD6789, DL3C1234, 22BH1234A)
        ind_patterns = [
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$',
            r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$',
            r'^[A-Z]{2}[0-9]{1,2}[0-9]{4}$'
        ]
        for pat in ind_patterns:
            if re.match(pat, text):
                return True
            
        # Flexible match for short/partial or numeric-only plates (3 to 10 alphanumeric characters)
        if 3 <= len(text) <= 10 and re.match(r'^[A-Z0-9]+$', text):
            return True
            
        return False

    def normalize_plate_characters(self, text: str) -> str:
        """
        Normalizes common OCR confusions in vehicle license plates:
        Replaces letter 'O' with digit '0', 'I' with '1', etc. in numeric positions.
        """
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not cleaned:
            return ""
        # If standard Indian format: 2 letters, 2 digits, 1-2 letters, 4 digits
        if len(cleaned) in [9, 10]:
            char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
            num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
            chars = list(cleaned)
            # State code: first 2 should be letters
            for i in range(min(2, len(chars))):
                if chars[i] in num_to_char:
                    chars[i] = num_to_char[chars[i]]
            # District code: chars 2,3 should be digits
            for i in range(2, min(4, len(chars))):
                if chars[i] in char_to_num:
                    chars[i] = char_to_num[chars[i]]
            # Suffix: last 4 should be digits
            for i in range(max(4, len(chars) - 4), len(chars)):
                if chars[i] in char_to_num:
                    chars[i] = char_to_num[chars[i]]
            return "".join(chars)
        return cleaned

    def clean_plate_text(self, text: str) -> str:
        # Keep only uppercase alphanumeric characters
        return re.sub(r'[^A-Z0-9]', '', text.upper())
        
    def read_plate(self, img) -> dict:
        """
        Runs multi-pass OCR on raw and CLAHE-enhanced plate crop.
        Returns a dict: {'raw_text': str, 'normalized_text': str, 'confidence': float, 'is_valid': bool}
        """
        if not self.enabled or img is None or img.size == 0:
            return None
            
        try:
            h, w = img.shape[:2]
            # Multi-pass: test enhanced upscaled image first, then raw image
            passes = []
            
            # Pass 1: Upscaled + CLAHE contrast enhancement (converted to 3-channel BGR for EasyOCR)
            scale = max(90.0 / max(h, 1), 2.5)
            if scale > 1.0:
                upscaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                upscaled = img.copy()
                
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Pass 2: Otsu Binary Thresholding (converted to 3-channel BGR)
            _, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_bgr = cv2.cvtColor(otsu_gray, cv2.COLOR_GRAY2BGR)
            
            passes.append(enhanced_bgr)
            passes.append(otsu_bgr)
            passes.append(upscaled)
            passes.append(img)
            
            best_candidate = None
            best_score = -1.0
            
            for pass_img in passes:
                results = self.ocr.readtext(pass_img)
                if not results:
                    continue
                    
                # Collect and sort text tokens horizontally
                tokens = []
                for (bbox, text, prob) in results:
                    cleaned = self.clean_plate_text(text)
                    if len(cleaned) >= 2 and prob >= 0.10:
                        x_center = (bbox[0][0] + bbox[1][0]) / 2.0
                        tokens.append((x_center, cleaned, prob))
                        
                if tokens:
                    tokens.sort(key=lambda t: t[0])
                    combined_text = "".join(t[1] for t in tokens)
                    normalized = self.normalize_plate_characters(combined_text)
                    avg_prob = sum(t[2] for t in tokens) / len(tokens)
                    is_ind = self.validate_indian_plate(normalized)
                    
                    # Score: Indian plates preferred, then longer valid strings
                    score = (100.0 if is_ind else 0.0) + (len(normalized) * 5.0) + avg_prob
                    if score > best_score and len(normalized) >= 3:
                        best_score = score
                        best_candidate = {
                            'raw_text': combined_text,
                            'normalized_text': normalized,
                            'confidence': float(round(avg_prob, 2)),
                            'is_valid': is_ind
                        }
                        
            return best_candidate
            
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return None
