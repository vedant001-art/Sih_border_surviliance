import time
from typing import Dict, Any, Optional

class TemporalANPRFusion:
    def __init__(self, cooldown_seconds: float = 0.3, max_history: int = 15):
        # track_id -> list of observation dicts
        self.history: Dict[int, list] = {}
        # track_id -> dict with best plate info
        self.best_plates: Dict[int, dict] = {}
        # track_id -> last OCR timestamp
        self.last_ocr_time: Dict[int, float] = {}
        
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history

    def can_run_ocr(self, track_id: int) -> bool:
        """Checks if OCR cooldown has expired for this track."""
        now = time.time()
        last = self.last_ocr_time.get(track_id, 0.0)
        return (now - last) >= self.cooldown_seconds

    def add_observation(self, track_id: int, plate_data: dict, plate_crop) -> Optional[dict]:
        """
        Adds an OCR observation and recalculates the fused best plate.
        plate_data: {"raw_text": str, "normalized_text": str, "confidence": float, "is_valid": bool}
        Returns the updated best plate dictionary.
        """
        self.last_ocr_time[track_id] = time.time()
        
        if track_id not in self.history:
            self.history[track_id] = []
            
        # Add to history
        self.history[track_id].append({
            "data": plate_data,
            "crop": plate_crop,
            "time": time.time()
        })
        
        # Keep only recent history
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id] = self.history[track_id][-self.max_history:]
            
        return self._fuse_observations(track_id)
        
    def _fuse_observations(self, track_id: int) -> dict:
        """
        Aggregates history to find the most likely plate string.
        Prioritizes valid Indian plates, then observation count, then confidence.
        """
        history = self.history[track_id]
        if not history:
            return {}
            
        # Group by normalized text
        groups = {}
        for obs in history:
            text = obs["data"]["normalized_text"]
            if text not in groups:
                groups[text] = {
                    "count": 0,
                    "max_conf": 0.0,
                    "is_valid": obs["data"]["is_valid"],
                    "best_crop": obs["crop"],
                    "best_raw": obs["data"]["raw_text"]
                }
            
            groups[text]["count"] += 1
            if obs["data"]["confidence"] > groups[text]["max_conf"]:
                groups[text]["max_conf"] = obs["data"]["confidence"]
                groups[text]["best_crop"] = obs["crop"]
                groups[text]["best_raw"] = obs["data"]["raw_text"]
                
        # Scoring: (is_valid * 100) + (count * 10) + max_conf
        best_text = None
        best_score = -1
        
        for text, stats in groups.items():
            score = (100 if stats["is_valid"] else 0) + (stats["count"] * 10) + stats["max_conf"]
            if score > best_score:
                best_score = score
                best_text = text
                
        best_stats = groups[best_text]
        
        fused = {
            "normalized_text": best_text,
            "raw_text": best_stats["best_raw"],
            "confidence": best_stats["max_conf"],
            "is_valid": best_stats["is_valid"],
            "observation_count": sum(g["count"] for g in groups.values()),
            "best_crop": best_stats["best_crop"]
        }
        
        self.best_plates[track_id] = fused
        return fused
        
    def get_best_plate(self, track_id: int) -> Optional[dict]:
        return self.best_plates.get(track_id)
        
    def cleanup(self, active_track_ids: set):
        expired = [tid for tid in self.history.keys() if tid not in active_track_ids]
        for tid in expired:
            del self.history[tid]
            if tid in self.best_plates:
                del self.best_plates[tid]
            if tid in self.last_ocr_time:
                del self.last_ocr_time[tid]

anpr_fusion = TemporalANPRFusion()
