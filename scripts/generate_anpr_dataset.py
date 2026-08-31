import os
import cv2
import numpy as np
import time
from loguru import logger
import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.detection.yolo_detector import YOLODetector
from ai.detection.plate_detector import PlateDetector
from ai.anpr.plate_reader import ANPRSystem

def calculate_laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

def create_yolo_label(x_min, y_min, x_max, y_max, img_width, img_height):
    # YOLO format: class_id x_center y_center width height (normalized)
    class_id = 0
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def main():
    parser = argparse.ArgumentParser(description="Generate ANPR Dataset from CCTV videos")
    parser.add_argument("--videos_dir", type=str, default="uploads", help="Directory containing MP4 videos")
    parser.add_argument("--out_dir", type=str, default="dataset/anpr", help="Output directory for dataset")
    parser.add_argument("--max_videos", type=int, default=5, help="Limit number of videos to process")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames to extract overall")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    
    # Create YOLO directory structure
    images_dir = out_dir / "images" / "train"
    labels_dir = out_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Write classes.txt
    with open(out_dir / "classes.txt", "w") as f:
        f.write("license_plate\n")

    logger.info("Initializing models...")
    vehicle_detector = YOLODetector()
    plate_detector = PlateDetector()
    ocr_system = ANPRSystem()

    videos = list(videos_dir.glob("*.mp4"))
    videos = videos[:args.max_videos]

    if not videos:
        logger.error(f"No videos found in {videos_dir}")
        return

    extracted_count = 0
    # Keep track of vehicles we've already extracted to avoid duplicates
    # We will reset this per video
    
    stats = {
        "distant": 0,
        "low_light": 0,
        "blurry": 0,
        "successful_ocr": 0,
        "failed_ocr": 0
    }

    for video_path in videos:
        if extracted_count >= args.max_frames:
            break
            
        logger.info(f"Processing video: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        
        extracted_tracks = set()
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Skip frames to speed up processing (process 1 in 5)
            if frame_idx % 5 != 0:
                continue

            h, w = frame.shape[:2]
            
            # Detect vehicles
            detections = vehicle_detector.detect_and_track(frame)
            
            for det in detections:
                if det["class_id"] not in [2, 3, 5, 7]: # Only vehicles
                    continue
                    
                track_id = det.get("track_id")
                if not track_id or track_id in extracted_tracks:
                    continue
                    
                # Extract crop
                vx1, vy1, vx2, vy2 = map(int, det["bbox"])
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w, vx2), min(h, vy2)
                
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                # Detect plate
                plate_res = plate_detector.detect_in_crop(vehicle_crop, vx1, vy1)
                
                if plate_res:
                    px1, py1, px2, py2 = plate_res["bbox"]
                    plate_crop = plate_res["crop"]
                    
                    # Try OCR
                    ocr_res = ocr_system.read_plate(plate_crop)
                    
                    # Categorize image
                    area = (px2 - px1) * (py2 - py1)
                    blur = calculate_laplacian_variance(plate_crop)
                    brightness = calculate_brightness(plate_crop)
                    
                    is_distant = area < 1000
                    is_blurry = blur < 100
                    is_lowlight = brightness < 50
                    
                    tags = []
                    if is_distant:
                        tags.append("distant")
                        stats["distant"] += 1
                    if is_blurry:
                        tags.append("blurry")
                        stats["blurry"] += 1
                    if is_lowlight:
                        tags.append("low_light")
                        stats["low_light"] += 1
                        
                    if ocr_res and ocr_res.get("is_valid"):
                        stats["successful_ocr"] += 1
                        tags.append("ocr_pass")
                    else:
                        stats["failed_ocr"] += 1
                        tags.append("ocr_fail")

                    # Save full frame (so it's context-aware)
                    img_name = f"{video_path.stem}_T{track_id}_{frame_idx}.jpg"
                    img_path = images_dir / img_name
                    cv2.imwrite(str(img_path), frame)
                    
                    # Save label
                    label_name = f"{video_path.stem}_T{track_id}_{frame_idx}.txt"
                    label_path = labels_dir / label_name
                    yolo_label = create_yolo_label(px1, py1, px2, py2, w, h)
                    
                    with open(label_path, "w") as f:
                        f.write(yolo_label + "\n")
                        
                    extracted_tracks.add(track_id)
                    extracted_count += 1
                    
                    if extracted_count >= args.max_frames:
                        break
                        
        cap.release()

    # Generate Report
    report = f"""# ANPR Dataset Generation Report

## Summary
- **Total Frames Extracted**: {extracted_count}
- **Output Directory**: `{out_dir.absolute()}`
- **Ready for Annotation**: Yes (Contains YOLO format images and labels)

## Breakdown of Extracted Frames
- **Successful OCR (Current System)**: {stats['successful_ocr']}
- **Failed OCR / Hard Examples**: {stats['failed_ocr']}

## Edge Case Categories
- **Distant/Small Plates**: {stats['distant']}
- **Low Light/Night**: {stats['low_light']}
- **Blurry**: {stats['blurry']}

## Next Steps
1. Use an annotation tool (like CVAT or LabelImg) to verify/adjust the bounding boxes in `{out_dir}/labels/train/`.
2. Split a percentage of files into `val/` and `test/` directories.
3. Use the `scripts/benchmark_anpr.py` script to benchmark the newly annotated dataset.
"""
    with open(out_dir / "dataset_report.md", "w") as f:
        f.write(report)
        
    logger.info(f"Dataset generated. Extracted {extracted_count} frames. Report saved to {out_dir}/dataset_report.md")

if __name__ == "__main__":
    main()
