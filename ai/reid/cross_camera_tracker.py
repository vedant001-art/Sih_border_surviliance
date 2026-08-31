import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from loguru import logger
import cv2

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Using ResNet18 as a lightweight ReID baseline for prototype
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the classification head
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"ReID Feature Extractor loaded on {self.device}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return None
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(tensor)
            features = features.cpu().numpy().flatten()
            # Normalize
            return features / np.linalg.norm(features)
        except Exception as e:
            logger.error(f"ReID Extractor error: {e}")
            return None

class GlobalTracker:
    def __init__(self, similarity_threshold: float = 0.75):
        self.extractor = FeatureExtractor()
        self.threshold = similarity_threshold
        # {global_track_id: embedding}
        self.global_tracks = {}
        self.next_global_id = 1
        
        # Mapping from (camera_id, local_track_id) -> global_track_id
        self.local_to_global = {}

    def get_global_id(self, camera_id: str, local_track_id: int, crop: np.ndarray) -> int:
        local_key = (camera_id, local_track_id)
        if local_key in self.local_to_global:
            return self.local_to_global[local_key]
            
        embedding = self.extractor.extract(crop)
        if embedding is None:
            return None
            
        best_match_id = None
        best_sim = -1.0
        
        for g_id, g_emb in self.global_tracks.items():
            sim = np.dot(embedding, g_emb)
            if sim > best_sim:
                best_sim = sim
                best_match_id = g_id
                
        if best_match_id is not None and best_sim > self.threshold:
            # Match found
            self.local_to_global[local_key] = best_match_id
            # Optionally update the embedding with moving average
            self.global_tracks[best_match_id] = 0.9 * self.global_tracks[best_match_id] + 0.1 * embedding
            self.global_tracks[best_match_id] /= np.linalg.norm(self.global_tracks[best_match_id])
            logger.info(f"ReID Matched: {camera_id} Local {local_track_id} -> Global {best_match_id} (sim: {best_sim:.2f})")
            return best_match_id
        else:
            # Create new global track
            new_id = self.next_global_id
            self.next_global_id += 1
            self.global_tracks[new_id] = embedding
            self.local_to_global[local_key] = new_id
            logger.info(f"ReID New Global Track: {camera_id} Local {local_track_id} -> Global {new_id}")
            return new_id
