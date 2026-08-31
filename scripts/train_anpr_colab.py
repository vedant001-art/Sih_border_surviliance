# ==============================================================================
# 🚀 GOOGLE COLAB 1-CLICK ANPR TRAINING SCRIPT (YOLOv8) - FIXED
# ==============================================================================
# Instructions:
# 1. Open Google Colab (https://colab.research.google.com)
# 2. Go to: Runtime -> Change runtime type -> Select "T4 GPU" -> Click Save
# 3. Replace the cell content with this code and click Run.
# ==============================================================================

# Step 1: Mount Google Drive
from google.colab import drive
import os
import shutil

print("--- Step 1: Mounting Google Drive ---")
drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/SIH_Model_Saves'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✓ Model saves directory: {SAVE_DIR}")

# Step 2: Install dependencies
print("\n--- Step 2: Installing Dependencies ---")
os.system("pip install --quiet ultralytics roboflow")

# Step 3: Download Verified Public License Plate Dataset
print("\n--- Step 3: Downloading License Plate Dataset ---")
import os
from roboflow import Roboflow

os.chdir('/content')

rf = Roboflow(api_key="mJUn86YkWsBpCxhhuLSn")

# Public curated Roboflow Universe License Plate Dataset
try:
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    dataset = project.version(4).download("yolov8")
    data_yaml_path = f"{dataset.location}/data.yaml"
    print(f"✓ Dataset successfully downloaded to: {dataset.location}")
except Exception as e:
    print(f"Roboflow download encountered issue ({e}). Downloading direct verified zip...")
    os.system('curl -L -o /content/license_plates.zip "https://universe.roboflow.com/ds/7xV1yLw1v2?key=mJUn86YkWsBpCxhhuLSn"')
    os.system('unzip -q /content/license_plates.zip -d /content/dataset')
    data_yaml_path = "/content/dataset/data.yaml"

# Step 4: Fine-tune YOLOv8 on T4 GPU
print("\n--- Step 4: Training YOLOv8s on T4 GPU ---")
from ultralytics import YOLO

# Transfer learning starting from yolov8s
model = YOLO('yolov8s.pt')

results = model.train(
    data=data_yaml_path,
    epochs=50,               # 50 epochs for high convergence
    imgsz=640,
    batch=16,
    device=0,                # T4 GPU
    project=SAVE_DIR,
    name='anpr_yolov8_run',
    save=True
)

# Step 5: Save best weights as anpr_best.pt
print("\n--- Step 5: Finalizing Model ---")
best_model_source = f"{SAVE_DIR}/anpr_yolov8_run/weights/best.pt"
final_target = f"{SAVE_DIR}/anpr_best.pt"

if os.path.exists(best_model_source):
    shutil.copy2(best_model_source, final_target)
    print("=" * 60)
    print("🎉 TRAINING SUCCESSFUL!")
    print(f"Model saved to your Google Drive: {final_target}")
    print("=" * 60)
else:
    print(f"Training finished. Check weights in: {SAVE_DIR}/anpr_yolov8_run/weights/")
