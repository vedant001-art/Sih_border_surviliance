import hashlib
import random

class FaceReIDEngine:
    def __init__(self):
        # Deterministic catalogs mapped to hardcoded mock profiles
        self.catalog = {
            "00": {"name": "Officer R. Verma", "clearance": "LEVEL-3 AUTH", "clothing": "Navy Blue Uniform"},
            "01": {"name": "Priya Sharma", "clearance": "LEVEL-2 AUTH", "clothing": "White Shirt / Blue Denim"},
            "02": {"name": "Rahul Sharma (BOLO)", "clearance": "RESTRICTED", "clothing": "Black Hoodie"},
            "03": {"name": "Alex Vance", "clearance": "LEVEL-1 VISITOR", "clothing": "Grey Jacket / Khaki"},
        }
        
    def get_person_profile(self, track_id: int) -> dict:
        """Deterministically hashes the track_id to assign a stable identity profile"""
        hash_val = hashlib.md5(str(track_id).encode()).hexdigest()
        
        # 40% chance of being a known catalog person, 60% chance of being unknown
        mod_val = int(hash_val[-2:], 16) % 10
        if mod_val < 4:
            # Map to one of the 4 known profiles
            cat_id = f"0{mod_val}"
            profile = self.catalog[cat_id].copy()
            profile["activity"] = "Walking" if int(hash_val[-3], 16) % 2 == 0 else "Standing"
            return profile
        else:
            colors = ["Red Shirt", "Blue T-Shirt", "Black Jacket", "White Top", "Green Hoodie"]
            color = colors[int(hash_val[-4], 16) % len(colors)]
            return {
                "name": "Unknown",
                "clearance": "UNVERIFIED",
                "clothing": color,
                "activity": "Walking" if int(hash_val[-3], 16) % 2 == 0 else "Standing"
            }

reid_engine = FaceReIDEngine()
