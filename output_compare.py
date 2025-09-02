
import os
import cv2
import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.config_loader import ConfigParser
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------
# Config & Paths
# ------------------------------
config = ConfigParser().get_data()
paths = config.get("paths", {})
IMG_PATH = os.path.join(paths.get("output_labels"), "labels_filtered/val/images")
GT_PATH = os.path.join(paths.get("output_labels"), "labels/val")
PRED_PATH = os.path.join(paths.get("output_labels"), "labels_filtered/preds")

# Classes
CLASS_NAMES = [
    "person", "rider", "car", "bus", "truck", "bike",
    "motor", "traffic light", "traffic sign", "train"
]

TF = 0.5  # IoU threshold for true positive
TB = 0.1  # IoU threshold for background

# ------------------------------
# Utility functions
# ------------------------------
def read_boxes(txt_file, pred=False):
    try:
        with open(txt_file, "r") as f:
            data = f.readlines()
        boxes = []
        for line in data:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                conf = float(parts[5]) if pred and len(parts) == 6 else 1.0
                boxes.append([int(cls), x, y, w, h, conf])
        return np.array(boxes)
    except FileNotFoundError:
        return np.array([])
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return np.array([])

def compute_iou(box1, box2):
    """Compute IoU between two boxes [cls, x, y, w, h, conf]"""
    def to_xyxy(box):
        _, x, y, w, h, _ = box
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2
    x1_min, y1_min, x1_max, y1_max = to_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = to_xyxy(box2)
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    inter_area = max(0, inter_xmax-inter_xmin)*max(0, inter_ymax-inter_ymin)
    box1_area = (x1_max-x1_min)*(y1_max-y1_min)
    box2_area = (x2_max-x2_min)*(y2_max-y2_min)
    union = box1_area + box2_area - inter_area
    return inter_area/union if union>0 else 0.0

def load_dataset(img_folder=IMG_PATH):
    """Load all images with their GT and predictions"""
    dataset = []
    img_files = list(Path(img_folder).glob("*.jpg"))
    for img_file in tqdm(img_files, desc="Loading dataset"):
        img_id = img_file.stem
        gt_boxes = read_boxes(os.path.join(GT_PATH, f"{img_id}.txt"))
        pred_boxes = read_boxes(os.path.join(PRED_PATH, f"{img_id}.txt"), pred=True)
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2] if img is not None else (0,0)
        dataset.append({
            "image_id": img_id,
            "image_path": str(img_file),
            "w": w,
            "h": h,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes
        })
    return dataset

# ------------------------------
# Main analysis
# ------------------------------
def analyze_dataset(dataset, out_csv="detection_analysis.csv"):
    rows = []
    for sample in tqdm(dataset, desc="Analyzing"):
        img_id = sample["image_id"]
        w, h = sample["w"], sample["h"]
        gt_boxes = sample["gt_boxes"]
        preds = sorted(sample["pred_boxes"], key=lambda x:-x[5])  # sort by confidence

        matched_gt = set()
        tp, fp, fn = 0, 0, 0

        for pred in preds:
            pred_cls, px, py, pw, ph, conf = pred
            best_iou, best_gt = 0, None
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(gt, pred)
                if iou > best_iou:
                    best_iou, best_gt = iou, i

            # Determine error type
            error_type = "Background Error"
            if best_gt is not None:
                gt_cls, gx, gy, gw, gh, _ = gt_boxes[best_gt]
                if best_iou >= TF:
                    if pred_cls == gt_cls:
                        if best_gt in matched_gt:
                            error_type = "Duplicate Detection Error"
                        else:
                            tp += 1
                            matched_gt.add(best_gt)
                            error_type = "Correct"
                    else:
                        error_type = "Classification Error"
                elif TB <= best_iou < TF:
                    if pred_cls == gt_cls:
                        error_type = "Localization Error"
                    else:
                        error_type = "Both Cls and Loc Error"
                else:
                    error_type = "Background Error"
            else:
                fp +=1

            rows.append({
                "image_id": img_id,
                "image_path": sample["image_path"],
                "w": w,
                "h": h,
                "gt_x_center": gx if best_gt is not None else None,
                "gt_y_center": gy if best_gt is not None else None,
                "gt_w": gw if best_gt is not None else None,
                "gt_h": gh if best_gt is not None else None,
                "pred_x_center": px,
                "pred_y_center": py,
                "pred_w": pw,
                "pred_h": ph,
                "conf_score": conf,
                "precision": tp/(tp+fp+1e-6),
                "recall": tp/(tp+fn+1e-6),
                "f1": 2*tp/(2*tp+fp+fn+1e-6),
                "error_type": error_type,
                "gt_class": int(gt_cls) if best_gt is not None else None,
                "pred_class": int(pred_cls),
                "gt_class_name": CLASS_NAMES[int(gt_cls)] if best_gt is not None else None,
                "pred_class_name": CLASS_NAMES[int(pred_cls)]
            })

        # Handle missed GT
        missed = set(range(len(gt_boxes))) - matched_gt
        for mid in missed:
            gt_cls, gx, gy, gw, gh, _ = gt_boxes[mid]
            fn +=1
            rows.append({
                "image_id": img_id,
                "image_path": sample["image_path"],
                "w": w,
                "h": h,
                "gt_x_center": gx,
                "gt_y_center": gy,
                "gt_w": gw,
                "gt_h": gh,
                "pred_x_center": None,
                "pred_y_center": None,
                "pred_w": None,
                "pred_h": None,
                "conf_score": None,
                "precision": tp/(tp+fp+1e-6),
                "recall": tp/(tp+fn+1e-6),
                "f1": 2*tp/(2*tp+fp+fn+1e-6),
                "error_type": "Missed GT Error",
                "gt_class": int(gt_cls),
                "pred_class": None,
                "gt_class_name": CLASS_NAMES[int(gt_cls)],
                "pred_class_name": None
            })

    # Save CSV
    keys = rows[0].keys()
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved analysis to {out_csv}")


# ------------------------------
# Function to generate COCO results
# ------------------------------
def save_coco_results(dataset, out_json="coco_results.json"):
    images = []
    anns = []
    preds = []
    ann_id = 1

    for idx, sample in enumerate(dataset):
        img_id = idx + 1
        images.append({
            "id": img_id,
            "file_name": sample["image_id"] + ".jpg",
            "width": sample["w"],
            "height": sample["h"]
        })

        # GT annotations
        for b in sample["gt_boxes"]:
            cls, x, y, w, h, _ = b
            x_min = (x - w/2) * sample["w"]
            y_min = (y - h/2) * sample["h"]
            w_box = w * sample["w"]
            h_box = h * sample["h"]
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x_min, y_min, w_box, h_box],
                "area": w_box*h_box,
                "iscrowd": 0
            })
            ann_id += 1

        # Predictions
        for b in sample["pred_boxes"]:
            cls, x, y, w, h, conf = b
            x_min = (x - w/2) * sample["w"]
            y_min = (y - h/2) * sample["h"]
            w_box = w * sample["w"]
            h_box = h * sample["h"]
            preds.append({
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x_min, y_min, w_box, h_box],
                "score": conf
            })

    # Save JSON
    coco_output = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)],
        "predictions": preds
    }

    with open(out_json, "w") as f:
        json.dump(coco_output, f, indent=4)
    print(f"Saved COCO results to {out_json}")

# ------------------------------
# Run
# ------------------------------
if __name__=="__main__":
    dataset = load_dataset()
    analyze_dataset(dataset, out_csv="detection_analysis.csv")
    save_coco_results(dataset, out_json="coco_results.json")
