import requests
import json
from loguru import logger
from backend.core.config import settings

class MistralService:
    def __init__(self):
        self.api_key = settings.MISTRAL_API_KEY
        self.model = settings.MISTRAL_MODEL
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.enabled = bool(self.api_key and self.api_key != "your_mistral_api_key_here")

    def generate_event_summary(self, event_data: dict) -> dict:
        """
        Takes structured event data and returns a dict with 'summary' and 'source'.
        Does not crash the pipeline if it fails.
        """
        if not self.enabled:
            return {"summary": self._deterministic_fallback(event_data), "source": "template"}
            
        prompt = f"""
        You are an AI assistant for a professional border surveillance system.
        Summarize the following verified event data into a concise, professional, single-sentence summary (max 2 sentences).
        Do not introduce facts not present in the data.
        
        Event Data:
        {json.dumps(event_data, indent=2)}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise surveillance AI summarizer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                summary = response.json()["choices"][0]["message"]["content"].strip()
                return {"summary": summary, "source": "mistral"}
            else:
                logger.error(f"Mistral API error: {response.text}")
                return {"summary": self._deterministic_fallback(event_data), "source": "template"}
        except Exception as e:
            logger.error(f"Mistral API exception: {e}")
            return {"summary": self._deterministic_fallback(event_data), "source": "template"}

    def _deterministic_fallback(self, event_data: dict) -> str:
        # If explicit informative message already provided, use it directly
        if event_data.get("message"):
            return event_data["message"]
            
        cam = event_data.get("camera_id", "Unknown Camera")
        loc = event_data.get("location", cam)
        evt_type = event_data.get("type") or event_data.get("event_type", "INTRUSION")
        tid = event_data.get("local_track_id") or event_data.get("track_id", "Unknown")
        obj_type = event_data.get("object_type", "target")
        cls_name = (event_data.get("class_name") or obj_type).upper()
        details = event_data.get("details", {})
        zone = details.get("zone_name", f"Restricted Sector {cam}")
        
        if obj_type == "person":
            clothing = details.get("clothing", "")
            clearance = details.get("clearance", "UNVERIFIED")
            desc = f" ({clothing}, {clearance})" if clothing else ""
            return f"PERIMETER BREACH: Person #{tid}{desc} entered {zone}."
        else:
            plate = details.get("plate") or event_data.get("plate", "UNREGISTERED")
            return f"PERIMETER BREACH: {cls_name} #{tid} (Plate: {plate}) entered {zone}."

mistral_service = MistralService()
