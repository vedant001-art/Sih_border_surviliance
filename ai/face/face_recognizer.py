import cv2
import numpy as np
from loguru import logger
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None
    logger.warning("InsightFace not installed. Face recognition disabled.")

class FaceRecognizer:
    _shared_known_faces = {}
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.app = None
        if FaceAnalysis:
            # Initialize InsightFace with buffalo_l (includes SCRFD and ArcFace)
            self.app = FaceAnalysis(name='buffalo_l', root='~/.insightface')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace FaceAnalysis initialized")

    @property
    def known_faces(self):
        return self._shared_known_faces

    def enroll(self, name: str, image: np.ndarray) -> bool:
        """Enroll a face from an image."""
        if not self.app:
            return False
            
        faces = self.app.get(image)
        if len(faces) == 0:
            logger.warning(f"No faces found for enrollment of {name}")
            return False
            
        # Assume the largest face is the target
        largest_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        self.known_faces[name] = largest_face.normed_embedding
        logger.info(f"Enrolled face for {name}")
        return True

    def recognize(self, frame: np.ndarray) -> list:
        """
        Detect and recognize faces in the frame.
        Returns: [{"bbox": [x1,y1,x2,y2], "name": str, "confidence": float}, ...]
        """
        if not self.app or frame is None:
            return []
            
        faces = self.app.get(frame)
        results = []
        
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            embedding = face.normed_embedding
            
            best_match = "Unknown"
            best_sim = -1.0
            
            for name, known_emb in self.known_faces.items():
                # Cosine similarity
                sim = np.dot(embedding, known_emb)
                if sim > best_sim:
                    best_sim = float(sim)
                    best_match = name
                    
            if best_sim < self.threshold:
                best_match = "Unknown"
                
            results.append({
                "bbox": bbox,
                "name": best_match,
                "confidence": best_sim
            })
            
        return results
