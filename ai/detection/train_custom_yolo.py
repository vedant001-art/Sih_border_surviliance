# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import os

def train_border_security_model():
    """
    Script to train a custom YOLOv8 model for Border Security.
    Requires a data.yaml file pointing to your custom thermal/drone/occluded dataset.
    """
    print("Initializing YOLOv8 training for Border Security...")
    
    # 1. Load the pre-trained model as a starting point (transfer learning)
    # Using 'yolov8s.pt' (Small) is recommended for real-time video processing. 
    # Use 'yolov8m.pt' (Medium) if you need higher accuracy and have a strong GPU.
    model = YOLO("yolov8s.pt")
    
    # 2. Check if dataset config exists
    data_yaml = "data.yaml"
    if not os.path.exists(data_yaml):
        print(f"ERROR: Could not find {data_yaml}!")
        print("Please ensure your dataset is in YOLOv8 format and you have a data.yaml file.")
        print("Example data.yaml contents:")
        print("  train: ./dataset/images/train")
        print("  val: ./dataset/images/val")
        print("  nc: 2")
        print("  names: ['person', 'vehicle']")
        return

    # 3. Start Training
    # epochs=100 is usually a good starting point to see if the model learns.
    # imgsz=640 is standard, but use 1280 if your drone cars are tiny!
    print("Starting training (this may take several hours depending on your GPU)...")
    results = model.train(
        data=data_yaml,
        epochs=100,           # Number of times to loop over the dataset
        imgsz=640,            # Image resolution
        batch=16,             # Batch size (lower this to 8 if you get Out of Memory errors)
        device=0,             # Use GPU 0 (or 'cpu' if no GPU, but that will take forever)
        name="border_model",  # Name of the output folder
        patience=20           # Early stopping if no improvement after 20 epochs
    )
    
    print("\nTraining Complete!")
    print("Your new custom model is saved in the 'runs/detect/border_model/weights/' directory.")
    print("Look for 'best.pt' and copy it to your project to use it!")

if __name__ == "__main__":
    train_border_security_model()
