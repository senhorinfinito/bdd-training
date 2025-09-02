# Results Dashboard

The **Results Dashboard** is designed to analyze and visualize the performance of the trained YOLOv11m model on the validation set of the BDD100K dataset. It provides both quantitative error analysis and interactive visual exploration.

---

## 1. Pipeline Overview

1. **Inference**  
   - The model generates predictions on the validation split.  
   - Results are stored in:  
     ```
     labels_filtered/preds/
     ```

2. **Comparison Script** (`output_compare.py`)  
   - Matches predictions against ground truth.  
   - Categorizes detections into:
     - Correct Detections
     - Classification Errors
     - Localization Errors
     - Duplicate Detections
     - Missed Ground Truth  
   - Produces two outputs:
     - `detection_analysis.csv` → Tabular error-level analysis.  
     - `coco_results.json` → COCO-style formatted results.  

3. **Visualization App** (`output_vis.py`)  
   - Built using Streamlit.  
   - Loads the comparison outputs.  
   - Provides dashboards for error analysis, per-class performance, and sample-based exploration.  

---

## 2. Dashboard Features

- **Error Distribution**  
  Visualize the frequency of different error types (e.g., classification vs. localization).  

- **Per-Class Metrics**  
  Inspect precision, recall, and F1-scores for each object category.  

- **Image-Level Drilldown**  
  Explore detections vs. ground truth bounding boxes on specific validation images.  

- **COCO Evaluation Compatibility**  
  The exported JSON can be directly used with COCO tools (`pycocotools`) for mAP/mAR evaluation.  

---

## 3. How to Use

1. Run inference with YOLOv11m → predictions saved under `labels_filtered/preds/`.  
2. Run the comparison script:  
   ```bash
   python3 output_compare.py
    ````

This generates `detection_analysis.csv` and `coco_results.json`.

3. Launch the dashboard:

   ```bash
   streamlit run output_vis.py
   ```

4. Interactively explore model results and error patterns.

---

## 4. Insights & Next Steps

* The dashboard highlights both **systematic issues** (e.g., class imbalance, small object misses) and **random errors** (e.g., ambiguous occlusions).
* These insights can guide further data filtering, augmentation strategies, and model tuning.
* Future improvements may include:

  * Adding trend plots across training epochs.
  * Integrating confidence threshold analysis.
  * Overlaying side-by-side GT vs. predictions for bulk samples.

---

**Note:** This dashboard is complementary to the dataset insights (`docs/dataset.md`) and training details (`docs/yolo_training.md`). Together, they provide a complete picture of model development, evaluation, and iteration.

## Error Types in the Results Dashboard

The dashboard provides a structured breakdown of detection outcomes into six main error types.  
This categorization follows standard definitions (similar to COCO analysis), making it easier to understand where the model fails.

1. **Classification Error**  
   IoU ≥ τf with a ground truth of the **incorrect class**.  
   → Correct localization but wrong label.

2. **Localization Error**  
   τb ≤ IoU < τf with the ground truth of the **correct class**.  
   → Right class, but bounding box not accurate enough.

3. **Both Classification & Localization Error**  
   τb ≤ IoU < τf with the ground truth of the **incorrect class**.  
   → Wrong class and poorly localized.

4. **Duplicate Detection Error**  
   IoU ≥ τf with the correct class, but the ground truth is already matched with a higher-scoring detection.  
   → Extra bounding box for the same object.

5. **Background Error**  
   IoU < τb for all ground truths.  
   → Model predicts object where none exists (false positive).

6. **Missed GT Error**  
   Ground truth object not matched with any prediction.  
   → False negative, object not detected.

---

These categories are directly computed during evaluation and are available as filters in the dashboard, enabling detailed inspection of model performance across images and classes.

### [Paper](https://dbolya.com/tide/paper.pdf)