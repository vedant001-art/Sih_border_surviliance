from typing import Dict, List

class RiskEngine:
    def __init__(self):
        # Base weights for different event types
        self.weights = {
            "INTRUSION": 70,
            "PREDICTED_INTRUSION": 50,
            "LOITERING": 40,
            "ABANDONED_OBJECT": 80,
            "VEHICLE_STOPPING": 60,
            "FACE_MATCH_KNOWN": 20, # E.g. friendly
            "FACE_MATCH_UNKNOWN": 40,
            "ANPR_MATCH": 30
        }
        
    def calculate_risk(self, events: List[Dict], context: Dict = None) -> float:
        """
        Calculates a 0-100 risk score based on a list of events related to an incident or track.
        """
        if not events:
            return 0.0
            
        total_score = 0
        for event in events:
            event_type = event.get("type", "UNKNOWN")
            base_score = self.weights.get(event_type, 10)
            
            # Apply multipliers based on context (e.g. night time)
            multiplier = 1.0
            if context:
                if context.get("is_night", False):
                    multiplier *= 1.2
                if context.get("restricted_zone", False):
                    multiplier *= 1.5
                    
            total_score += (base_score * multiplier)
            
        # Cap at 100
        return min(100.0, total_score)
