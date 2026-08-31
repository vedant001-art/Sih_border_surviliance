import os
import argparse
import time
import json
from pathlib import Path
from loguru import logger
from ultralytics import YOLO

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.anpr.plate_reader import ANPRSystem

def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def parse_yolo_label(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class_id, x_c, y_c, w, h
                x_c, y_c, w, h = map(float, parts[1:5])
                x1 = (x_c - w/2) * img_w
                y1 = (y_c - h/2) * img_h
                x2 = (x_c + w/2) * img_w
                y2 = (y_c + h/2) * img_h
                boxes.append([x1, y1, x2, y2])
    return boxes

def main():
    parser = argparse.ArgumentParser(description="Benchmark ANPR Models")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to YOLO formatted dataset directory")
    parser.add_argument("--model", type=str, default="models/best_epoch22_undertrained.pt", help="Path to YOLO model weights to evaluate")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save_failures", action="store_true", help="Save crops where model fails")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    images_dir = dataset_path / "images" / "train"
    labels_dir = dataset_path / "labels" / "train"
    
    if args.save_failures:
        failure_dir = dataset_path / "failures" / Path(args.model).stem
        failure_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return

    logger.info(f"Loading YOLO Model from {args.model}")
    model = YOLO(args.model)
    ocr = ANPRSystem()

    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    stats = {
        "total_images": len(images),
        "total_gt_plates": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "successful_ocr": 0,
        "total_latency_ms": 0
    }

    import cv2
    
    logger.info(f"Starting evaluation on {len(images)} images...")
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Ground Truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path, w, h)
        stats["total_gt_plates"] += len(gt_boxes)
        
        # Inference
        start_time = time.time()
        results = model.predict(img, conf=args.conf_thresh, iou=args.iou_thresh, verbose=False)
        pred_boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy():
                pred_boxes.append(box.tolist())
                
        latency = (time.time() - start_time) * 1000
        stats["total_latency_ms"] += latency
        
        # Match Predictions to Ground Truth
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            if best_iou > 0.5:
                stats["true_positives"] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                
                # Test OCR on true positive crops
                x1, y1, x2, y2 = map(int, p_box)
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                ocr_res = ocr.read_plate(crop)
                if ocr_res and ocr_res.get("is_valid"):
                    stats["successful_ocr"] += 1
                else:
                    if args.save_failures:
                        cv2.imwrite(str(failure_dir / f"{img_path.stem}_ocr_fail.jpg"), crop)
            else:
                stats["false_positives"] += 1
                if args.save_failures:
                    x1, y1, x2, y2 = map(int, p_box)
                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size > 0:
                        cv2.imwrite(str(failure_dir / f"{img_path.stem}_fp.jpg"), crop)
                
        stats["false_negatives"] += (len(gt_boxes) - len(matched_gt))
        
        # If there were GT boxes but no predictions, save full frame as False Negative
        if len(gt_boxes) > 0 and len(matched_gt) == 0 and args.save_failures:
            cv2.imwrite(str(failure_dir / f"{img_path.stem}_fn.jpg"), img)
            
    # Calculate Metrics
    tp = stats["true_positives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    ocr_acc = stats["successful_ocr"] / tp if tp > 0 else 0.0
    end_to_end_acc = stats["successful_ocr"] / stats["total_gt_plates"] if stats["total_gt_plates"] > 0 else 0.0
    
    avg_latency = stats["total_latency_ms"] / len(images) if len(images) > 0 else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    report = f"""# ANPR Benchmark Report: {Path(args.model).name}

## 1. Detection Metrics (@IoU>0.5)
- **Total Ground Truth Plates**: {stats['total_gt_plates']}
- **True Positives**: {tp}
- **False Positives**: {fp}
- **False Negatives**: {fn}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}

## 2. Recognition Metrics
- **Plates Detected**: {tp}
- **Successful OCR**: {stats['successful_ocr']}
- **OCR Accuracy on Detected Plates**: {ocr_acc:.4f}

## 3. System Metrics
- **End-to-End System Accuracy**: {end_to_end_acc:.4f}
- **Average Inference Latency**: {avg_latency:.2f} ms
- **FPS**: {fps:.2f}
"""

    report_path = dataset_path / f"benchmark_{Path(args.model).stem}.md"
    with open(report_path, "w") as f:
        f.write(report)
        
    print("\n" + "="*50)
    print(report)
    print("="*50)
    logger.info(f"Benchmark completed. Report saved to {report_path}")

if __name__ == "__main__":
    main()
