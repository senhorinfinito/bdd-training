# BDD Training

This repository provides scripts for preprocessing, training, and visualizing the **BDD100K** dataset using Python 3.11, YOLOv11, and Streamlit.  
The workflow is designed to be executed inside a `python:3.11` Docker container in interactive mode.

---

## Prerequisites

- Docker installed and running
- BDD100K dataset available locally
- Git
- Sufficient GPU resources for training

---

## Running Inside Docker

### 1. Start a Python 3.11 container
Run an interactive container and mount the repository and dataset into it:

```bash
docker run -it --gpus all --name bdd-training \
    -v $(pwd):/workspace \
    -v /path/to/bdd100k:/workspace/dataset \
    -p 8501:8501 \
    python:3.11 bash
````

Explanation:

* `-v $(pwd):/workspace` mounts the repository into the container.
* `-v /path/to/bdd100k:/workspace/dataset` mounts the dataset (update with your dataset path).
* `-p 8501:8501` exposes Streamlit on localhost:8501.
* `--gpus all` ensures GPU access inside the container.

Inside the container, move to the workspace:

```bash
cd /workspace
```

---

## Configuration (`cfg/default.yaml`)

This repository uses a central configuration file.
Update dataset paths inside `cfg/default.yaml` according to your environment.

Example configuration:

```yaml
paths:
  image_root: /workspace/dataset/images/100k
  det_train: /workspace/dataset/labels/bdd100k_labels_images_train.json
  det_val: /workspace/dataset/labels/bdd100k_labels_images_val.json
  vis_dir: ./cfg/data/insights
  output_labels: ../final_dataset
  
# classes
classes:
  - person
  - rider
  - car
  - bus
  - truck
  - bike
  - motor
  - traffic light
  - traffic sign
  - train

convert: 
  yolo: true
```

---

# Workflow

## 1. Data Preparation

Run preprocessing to filter noisy bounding boxes and generate YOLO-style labels:

```bash
python3 src/data_prep.py
```

Outputs will be stored in:

* `final_dataset/labels_filtered/` → filtered YOLO labels
* `cfg/data/insights/` → statistics and visual insights

---

# My Observations [→ Click here to visualize](docs/dataset.md)

1. Very Small Objects – pedestrians, traffic lights, and motorcycles are often annotated at very low resolutions (below 16–20px).
2. Inconsistent Annotations – some objects are labeled in one frame/image but missing in the next, especially in sequences.
3. Class Imbalance – categories like cars and trucks dominate, while trains, bicycles, and some pedestrian types are rare.
4. Incorrect or Ambiguous Labels – mislabeled bounding boxes (e.g., truck as bus) and ambiguous cases with occlusion/truncation.
5. Contextual Bias – dataset is heavily skewed toward daytime, clear-weather, urban scenes, with rare coverage of night, snow, or rural conditions.

**Note:** You can explore all these points interactively using the visualization dashboard.

---

## 2. Model Training (YOLOv11)

Train YOLOv11 on the preprocessed dataset:

```bash
yolo detect train \
    data=cfg/default.yaml \
    model=yolov11m.pt \
    epochs=50 \
    imgsz=640 \
    project=runs/train \
    name=bdd_yolov11
```

Key Notes:

* Uses **YOLOv11m** as the backbone.
* Trains for 50 epochs (adjustable).
* Results (checkpoints, metrics, logs) saved in `runs/train/bdd_yolov11/`.

**Reference:** Training command and configs provided in [`src/yolo_training.md`](docs/yolo_training.md)

---

## 3. Visualization & Results Dashboard

### Step 1: Compare predictions with ground truth

```bash
python3 src/output_compare.py
```

This generates:

* `detection_analysis.csv` → detailed per-class and per-image analysis
* `coco_results.json` → COCO-style evaluation results

### Step 2: Launch Streamlit dashboard

```bash
streamlit run src/output_vis.py --server.port 8501 --server.address 0.0.0.0
```

Open in browser: [http://localhost:8501](http://localhost:8501)

---

# Results & Insights

## Training Summary

The YOLOv11m model was trained on the filtered BDD100K dataset where very small bounding boxes for large objects were removed. Apart from this, no major preprocessing was done. The model was trained with default settings and evaluated on the validation split, producing detection results in `labels_filtered/preds/`.

## Results Dashboard

After inference, the results were compared against the ground truth using `output_compare.py`, generating both a detailed CSV (`detection_analysis.csv`) and a COCO-style JSON (`coco_results.json`). These outputs feed into `output_vis.py`, which powers the interactive Streamlit dashboard. The dashboard allows exploration of error types (e.g., misclassifications, localization errors, missed detections) and provides a deeper view of per-class and per-image performance.

**Reference:** Results exploration and error analysis in [`docs/results_dashboard.md`](docs/results_dashboard.md)

---

# Documentation Links

* **Dataset Preparation & Insights** → `docs/dataset.md`
* **YOLO Training Process** → `docs/yolo_training.md`
* **Results Summary** → `docs/results_insights.md`
* **Interactive Dashboard for Results** → `docs/results_dashboard.md`

---

## Results Dashboard

We provide an interactive dashboard for analyzing detection outputs.
It includes detailed error categorization, per-class metrics, and image-level visualizations.

[Click here to view the Results Dashboard documentation](docs/results_dashboard.md)

**Reference:** Error taxonomy derived from [TIDE paper](https://dbolya.com/tide/paper.pdf).

