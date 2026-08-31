import time
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
from loguru import logger
try:
    import Levenshtein
except Exception:
    class Levenshtein:
        @staticmethod
        def distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return Levenshtein.distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]


class GlobalEntity:
    def __init__(self, entity_id: str, type: str):
        self.entity_id = entity_id
        self.type = type
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.attributes = {}
        
    def update(self, attributes: dict):
        self.last_seen = time.time()
        self.attributes.update(attributes)

class EntityRegistry:
    def __init__(self, timeout_seconds=15.0):
        self.entities: Dict[str, GlobalEntity] = {}
        self.timeout_seconds = timeout_seconds
        
        # Track active cameras to avoid stale state
        self.active_camera_tracks: Dict[str, Dict[int, str]] = {} # camera_id -> local_track_id -> global_entity_id
        
        # Cooldowns to prevent DB spam
        self.db_cooldowns: Dict[str, float] = {}

    def get_or_create(self, camera_id: str, local_track_id: int, type: str) -> GlobalEntity:
        if camera_id not in self.active_camera_tracks:
            self.active_camera_tracks[camera_id] = {}
            
        cam_tracks = self.active_camera_tracks[camera_id]
        
        if local_track_id in cam_tracks:
            entity_id = cam_tracks[local_track_id]
            entity = self.entities.get(entity_id)
            if entity:
                entity.last_seen = time.time()
                return entity
                
        # Create new entity if it doesn't match an existing recently lost one (spatial-temporal logic can be added here)
        entity_id = f"{'VEH' if type == 'vehicle' else 'PER'}_{camera_id}_{local_track_id}_{int(time.time())}"
        entity = GlobalEntity(entity_id, type)
        self.entities[entity_id] = entity
        cam_tracks[local_track_id] = entity_id
        
        return entity

    def cleanup(self):
        now = time.time()
        expired = [eid for eid, ent in self.entities.items() if now - ent.last_seen > self.timeout_seconds]
        for eid in expired:
            del self.entities[eid]
            # Remove from camera mappings
            for cam_id in self.active_camera_tracks:
                self.active_camera_tracks[cam_id] = {k: v for k, v in self.active_camera_tracks[cam_id].items() if v != eid}

    def canonicalize_plate(self, new_plate: str, global_entity_id: str) -> str:
        """Uses Levenshtein edit distance to prevent OCR flickering from creating new records"""
        entity = self.entities.get(global_entity_id)
        if not entity:
            return new_plate
            
        known_plate = entity.attributes.get('plate')
        if not known_plate:
            return new_plate
            
        # If the new plate is very close to the known plate (edit distance <= 2), keep the known plate
        if Levenshtein.distance(new_plate, known_plate) <= 2:
            return known_plate
            
        return new_plate
        
    def can_write_db(self, action: str, entity_id: str, cooldown_duration=45.0) -> bool:
        key = f"{action}_{entity_id}"
        last_write = self.db_cooldowns.get(key, 0)
        if time.time() - last_write > cooldown_duration:
            self.db_cooldowns[key] = time.time()
            return True
        return False
        
    def get_active_count(self) -> dict:
        now = time.time()
        # Consider active if seen in last 2 seconds
        active_persons = sum(1 for e in self.entities.values() if e.type == 'person' and now - e.last_seen < 2.0)
        active_vehicles = sum(1 for e in self.entities.values() if e.type == 'vehicle' and now - e.last_seen < 2.0)
        return {
            "persons": active_persons,
            "vehicles": active_vehicles,
            "total_verified_persons": sum(1 for e in self.entities.values() if e.type == 'person'),
            "total_verified_vehicles": sum(1 for e in self.entities.values() if e.type == 'vehicle')
        }

entity_registry = EntityRegistry()
