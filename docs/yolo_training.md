
---

# YOLOv11 Training Pipeline

## 1. Objective

The goal of this training experiment was to evaluate the **YOLOv11m** model on the filtered dataset, focusing on improving detection accuracy for large-scale traffic scenes. Smaller bounding boxes were removed to simplify the first round of experiments.

---

## 2. Dataset Preparation

* **Source**: Custom dataset with `labels` in YOLO format.
* **Preprocessing**:

  * Removed very small bounding boxes for large object classes (to reduce noise).
  * Dataset split into `train`, `val`, and `test` sets.
* **Classes**: 10 categories
  (`person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`).

---

## 3. Model Configuration

* **Base model**: `yolov11m` (medium variant of YOLOv11).
* **Input resolution**: 640 × 640.
* **Training parameters**:

  * Optimizer: SGD with momentum
  * Learning rate: `1e-3` (cosine scheduler)
  * Batch size: 16
  * Epochs: 100
* **Augmentations applied**:

  * Random horizontal flip
  * Random scale & crop
  * Color jitter (HSV adjustments)

---

## 4. Training Command

Training was performed using the official YOLOv11 interface inside the container.

```bash
yolo detect train \
    model=yolov11m.pt \
    data=cfg/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    project=runs/train \
    name=exp \
    device 0,1,2,3 \
    pretrained=True
```

---

## 5. Training Process

1. Model initialized with pretrained COCO weights.
2. Training executed on **multi GPU** setup.
3. Validation loss and mAP monitored every epoch.
4. Early stopping enabled if no improvement in `mAP@0.5` for 15 epochs.

---

## 6. Outputs

* Trained weights stored under:
  `runs/train/exp/weights/best.pt`
* Validation predictions stored in:
  `labels_filtered/preds/` (used later for dashboard analysis).
* Logs and training curves auto-saved with **TensorBoard** support.

---

## 7. Next Steps

* Fine-tune using additional augmentations and keep small-object boxes for robustness.
* Compare `yolov11m` with lighter (`yolov11s`) and heavier (`yolov11l/x`) variants.
* Evaluate generalization on the test set with dashboard analysis.

