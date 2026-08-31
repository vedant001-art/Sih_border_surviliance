import cv2
import os
from loguru import logger
from datetime import datetime

class EvidenceCapture:
    def __init__(self, output_dir: str = "evidence"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_clip(self, camera_id: str, event_id: int, frames: list, fps: float = 30.0) -> str:
        """
        Saves a list of OpenCV frames to an MP4 file.
        Returns the path to the saved file.
        """
        if not frames:
            logger.warning(f"No frames provided for evidence {event_id}")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_evt_{event_id}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Get frame dimensions
        height, width, layers = frames[0].shape
        size = (width, height)
        
        try:
            # We use mp4v codec for standard mp4
            out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            for frame in frames:
                out.write(frame)
            out.release()
            
            logger.info(f"Saved evidence clip for event {event_id} at {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save evidence clip: {e}")
            return None
            
    def save_image(self, camera_id: str, event_id: int, frame) -> str:
        if frame is None:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_evt_{event_id}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save evidence image: {e}")
            return None
