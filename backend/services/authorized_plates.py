import os
import json
import re
from typing import List, Set
from loguru import logger

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
PLATES_FILE = os.path.join(DATA_DIR, "authorized_plates.json")

# Default whitelist samples
DEFAULT_PLATES = [
    "DL8CAF1234",
    "KA01MJ5050",
    "MH12DE5678",
    "UP16AB9999",
    "HR26DK8888",
    "DL3CC4001"
]

def normalize_plate(plate: str) -> str:
    """Removes spaces, hyphens, dots, and converts to uppercase for robust matching."""
    if not plate:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(plate)).upper()

class AuthorizedPlatesService:
    def __init__(self):
        self.plates: Set[str] = set()
        self._load()

    def _load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(PLATES_FILE):
            self.plates = set(DEFAULT_PLATES)
            self._save()
        else:
            try:
                with open(PLATES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.plates = set(normalize_plate(p) for p in data if p)
                logger.info(f"Loaded {len(self.plates)} authorized plates from storage.")
            except Exception as e:
                logger.error(f"Failed to read {PLATES_FILE}, using defaults: {e}")
                self.plates = set(DEFAULT_PLATES)

    def _save(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(PLATES_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.plates)), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save authorized plates: {e}")

    def is_authorized(self, plate_text: str) -> bool:
        """
        Checks if a plate text is authorized.
        Supports exact match, substring containment, OCR character substitution, and edit distance.
        """
        if not plate_text or plate_text in ["UNREADABLE", "UNKNOWN", "SCANNING...", "NONE"]:
            return False
            
        norm = normalize_plate(plate_text)
        if not norm or len(norm) < 3:
            return False

        # 1. Exact match against authorized plates set
        if norm in self.plates:
            return True

        # OCR character canonicalization: 0/O/D/Q, 1/I/L, 8/B, 5/S, 2/Z
        def canonicalize(s: str) -> str:
            return s.replace('O', '0').replace('D', '0').replace('Q', '0') \
                    .replace('I', '1').replace('L', '1') \
                    .replace('B', '8') \
                    .replace('S', '5') \
                    .replace('Z', '2')

        c_norm = canonicalize(norm)

        # 2. Check against each registered plate
        for auth in self.plates:
            if norm == auth:
                return True
            # Substring match (e.g. "3789" in "KA013789" or "DL8CAF1234" containing "8CAF1234")
            if len(auth) >= 4 and auth in norm:
                return True
            if len(norm) >= 4 and norm in auth:
                return True

            c_auth = canonicalize(auth)
            if c_norm == c_auth:
                return True
            if len(c_auth) >= 4 and (c_auth in c_norm or c_norm in c_auth):
                return True

            # Single character edit distance tolerance for OCR misread on longer plates
            if len(auth) == len(norm) and len(auth) >= 6:
                diffs = sum(1 for a, b in zip(auth, norm) if a != b)
                if diffs <= 1:
                    return True
        return False

    def get_all(self) -> List[str]:
        return sorted(list(self.plates))

    def add_plate(self, plate: str) -> bool:
        norm = normalize_plate(plate)
        if norm:
            self.plates.add(norm)
            self._save()
            logger.info(f"Added authorized plate: {norm}")
            return True
        return False

    def remove_plate(self, plate: str) -> bool:
        norm = normalize_plate(plate)
        if norm in self.plates:
            self.plates.remove(norm)
            self._save()
            logger.info(f"Removed authorized plate: {norm}")
            return True
        return False

    def set_plates(self, plates: List[str]):
        self.plates = set(normalize_plate(p) for p in plates if normalize_plate(p))
        self._save()
        logger.info(f"Set authorized plates registry to {len(self.plates)} entries.")

authorized_plates = AuthorizedPlatesService()
