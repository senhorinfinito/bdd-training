import os
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import altair as alt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# ------------------------------
# 1. Streamlit page config
# ------------------------------
st.set_page_config(layout="wide")
st.title("Detection Result Analysis Dashboard")

# ------------------------------
# 2. Sidebar: Load CSV
# ------------------------------
st.sidebar.header("Load Files")
uploaded_file = st.sidebar.file_uploader("Upload results CSV", type=["csv"], key="csv")
DEFAULT_CSV = "./detection_analysis.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV loaded")
elif os.path.exists(DEFAULT_CSV):
    df = pd.read_csv(DEFAULT_CSV)
    st.sidebar.info(f"Default CSV loaded: {DEFAULT_CSV}")
else:
    df = None
    st.sidebar.warning("No CSV file found")

# ------------------------------
# 3. Generate COCO format from CSV and evaluate
# ------------------------------
def generate_coco_from_csv(df):
    images = []
    anns = []
    preds = []
    ann_id = 1
    img_id_map = {}
    
    CLASS_NAMES = sorted(df["gt_class_name"].dropna().unique()) if "gt_class_name" in df.columns else []
    class_name_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    for idx, (image_id, group) in enumerate(df.groupby("image_id")):
        img_id = idx + 1
        img_id_map[image_id] = img_id
        w, h = int(group.iloc[0]["w"]), int(group.iloc[0]["h"])
        images.append({
            "id": img_id,
            "file_name": group.iloc[0]["image_path"].split("/")[-1],
            "width": w,
            "height": h
        })

        # GT Annotations
        for _, row in group.iterrows():
            if pd.notna(row.get("gt_x_center")):
                cls_name = row["gt_class_name"]
                cls_id = class_name_to_id[cls_name]
                x_center, y_center, bw, bh = row["gt_x_center"], row["gt_y_center"], row["gt_w"], row["gt_h"]
                x_min = (x_center - bw/2) * w
                y_min = (y_center - bh/2) * h
                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, bw*w, bh*h],
                    "area": bw*w * bh*h,
                    "iscrowd": 0
                })
                ann_id += 1

        # Predicted boxes
        for _, row in group.iterrows():
            if pd.notna(row.get("pred_x_center")):
                cls_name = row["pred_class_name"]
                cls_id = class_name_to_id.get(cls_name, 0)
                x_center, y_center, bw, bh, conf = row["pred_x_center"], row["pred_y_center"], row["pred_w"], row["pred_h"], row["conf_score"]
                x_min = (x_center - bw/2) * w
                y_min = (y_center - bh/2) * h
                preds.append({
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, bw*w, bh*h],
                    "score": conf
                })

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    
    # Add mandatory fields for pycocotools
    coco_gt_dict = {
        "info": {"description": "Generated from CSV", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": categories
    }

    return coco_gt_dict, preds

# Evaluate COCO metrics
coco_metrics_available = False
if df is not None:
    try:
        coco_gt_dict, coco_pred_list = generate_coco_from_csv(df)
        with open("temp_coco_gt.json", "w") as f:
            json.dump(coco_gt_dict, f)
        coco_gt = COCO("temp_coco_gt.json")
        coco_dt = coco_gt.loadRes(coco_pred_list)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics_available = True

        # Display metrics
        coco_stats = {
            "AP@[0.5:0.95]": coco_eval.stats[0],
            "AP@0.5": coco_eval.stats[1],
            "AP@0.75": coco_eval.stats[2],
            "AP (small)": coco_eval.stats[3],
            "AP (medium)": coco_eval.stats[4],
            "AP (large)": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],
            "AR@10": coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
            "AR (small)": coco_eval.stats[9],
            "AR (medium)": coco_eval.stats[10],
            "AR (large)": coco_eval.stats[11],
        }
        st.subheader("COCO Metrics")
        st.table(pd.DataFrame(coco_stats.items(), columns=["Metric", "Value"]))
    except Exception as e:
        st.error(f"COCO evaluation error: {e}")

# ------------------------------
# 4. CSV Analysis & Image Viewer
# ------------------------------
if df is not None:
    # Filters
    st.sidebar.header("Filters")
    gt_classes = ["All"] + sorted(df["gt_class_name"].dropna().unique()) if "gt_class_name" in df.columns else ["All"]
    selected_gt = st.sidebar.selectbox("GT Class", gt_classes)
    pred_classes = ["All"] + sorted(df["pred_class_name"].dropna().unique()) if "pred_class_name" in df.columns else ["All"]
    selected_pred = st.sidebar.selectbox("Pred Class", pred_classes)
    error_types = ["All"] + sorted(df["error_type"].dropna().unique()) if "error_type" in df.columns else ["All"]
    selected_error = st.sidebar.selectbox("Error Type", error_types)
    conf_range = (float(df["conf_score"].min()), float(df["conf_score"].max())) if "conf_score" in df.columns else (0.0, 1.0)
    if "conf_score" in df.columns:
        conf_range = st.sidebar.slider("Confidence range", min_value=conf_range[0], max_value=conf_range[1],
                                       value=conf_range, step=0.01)

    # Apply filters
    filtered = df.copy()
    if selected_gt != "All": filtered = filtered[filtered["gt_class_name"] == selected_gt]
    if selected_pred != "All": filtered = filtered[filtered["pred_class_name"] == selected_pred]
    if selected_error != "All": filtered = filtered[filtered["error_type"] == selected_error]
    if "conf_score" in df.columns:
        filtered = filtered[(filtered["conf_score"] >= conf_range[0]) & (filtered["conf_score"] <= conf_range[1])]

    # Sorting
    sort_by = st.sidebar.selectbox("Sort by", ["precision", "recall", "f1", "conf_score", "gt_w", "gt_h"], index=0)
    sort_order = st.sidebar.radio("Order", ["Descending", "Ascending"], index=0)
    filtered = filtered.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))

    # Stats Panel
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    if "precision" in filtered.columns: col1.metric("Avg Precision", f"{filtered['precision'].mean():.3f}")
    if "recall" in filtered.columns: col2.metric("Avg Recall", f"{filtered['recall'].mean():.3f}")
    if "f1" in filtered.columns: col3.metric("Avg F1", f"{filtered['f1'].mean():.3f}")

    # Error count chart
    if "error_type" in filtered.columns:
        error_counts = filtered["error_type"].value_counts().reset_index()
        error_counts.columns = ["error_type", "count"]
        chart = (alt.Chart(error_counts)
                 .mark_bar()
                 .encode(x="error_type:N", y="count:Q", color="error_type:N", tooltip=["error_type", "count"]))
        st.altair_chart(chart, use_container_width=True)

    # Image Viewer
    st.subheader("Image Viewer")
    ERROR_COLORS = {
        "Classification Error": "red",
        "Localization Error": "orange",
        "Both Cls and Loc Error": "purple",
        "Duplicate Detection Error": "blue",
        "Background Error": "brown",
        "Missed GT Error": "gray",
        "Correct": "green"
    }
    if "image_path" in filtered.columns:
        num_cols = 4
        cols = st.columns(num_cols)
        i = 0
        for image_id, group in filtered.groupby("image_id"):
            image_path = group.iloc[0]["image_path"]
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                draw = ImageDraw.Draw(img)
                w, h = img.size
                for _, row in group.iterrows():
                    # GT box
                    if pd.notna(row.get("gt_x_center")):
                        gt_x1 = row["gt_x_center"] * w - row["gt_w"] * w / 2
                        gt_y1 = row["gt_y_center"] * h - row["gt_h"] * h / 2
                        gt_x2 = row["gt_x_center"] * w + row["gt_w"] * w / 2
                        gt_y2 = row["gt_y_center"] * h + row["gt_h"] * h / 2
                        draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline="green", width=3)
                        draw.text((gt_x1, gt_y1 - 10), str(row["gt_class_name"]), fill="green")
                    # Pred box
                    if pd.notna(row.get("pred_x_center")):
                        pred_x1 = row["pred_x_center"] * w - row["pred_w"] * w / 2
                        pred_y1 = row["pred_y_center"] * h - row["pred_h"] * h / 2
                        pred_x2 = row["pred_x_center"] * w + row["pred_w"] * w / 2
                        pred_y2 = row["pred_y_center"] * h + row["pred_h"] * h / 2
                        color = ERROR_COLORS.get(row["error_type"], "yellow")
                        draw.rectangle([pred_x1, pred_y1, pred_x2, pred_y2], outline=color, width=3)
                        conf_text = f"{row['pred_class_name']} ({row['conf_score']:.2f})" if pd.notna(row['conf_score']) else str(row['pred_class_name'])
                        draw.text((pred_x1, pred_y1 - 10), conf_text, fill=color)

                # Render image
                with cols[i % num_cols]:
                    st.image(img, caption=f"{image_id}", use_container_width=True)
                i += 1
                if i % num_cols == 0:
                    cols = st.columns(num_cols)

    # Download filtered CSV
    st.subheader("Download Filtered Results")
    st.download_button(
        label="Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv"
    )
