

import os
import json
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import altair as alt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ------------------------------
# 1. Streamlit page config
# ------------------------------
st.set_page_config(layout="wide")
st.title("Detection Result Analysis Dashboard")

# ------------------------------
# 2. Sidebar: Load files
# ------------------------------
st.sidebar.header("Load Files")

# CSV file uploader
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

# COCO GT/Prediction JSON upload
uploaded_gt = st.sidebar.file_uploader("Upload GT JSON", type=["json"], key="gt")
uploaded_pred = st.sidebar.file_uploader("Upload Pred JSON", type=["json"], key="pred")
DEFAULT_GT = "./temp_gt.json"
DEFAULT_PRED = "./temp_pred.json"

gt_file = uploaded_gt if uploaded_gt else DEFAULT_GT
pred_file = uploaded_pred if uploaded_pred else DEFAULT_PRED

# ------------------------------
# 3. COCO evaluation
# ------------------------------
coco_metrics_available = False
if os.path.exists(gt_file) and os.path.exists(pred_file):
    try:
        coco_gt = COCO(gt_file)
        
        # Load predictions safely
        with open(pred_file) as f:
            preds = json.load(f)
        if isinstance(preds, dict) and "annotations" in preds:
            preds = preds["annotations"]
        coco_dt = coco_gt.loadRes(preds)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics_available = True

        # Prepare COCO stats table
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
# 4. CSV analysis (if CSV loaded)
# ------------------------------
if df is not None:
    # ------------------------------
    # Filters
    # ------------------------------
    st.sidebar.header("Filters")
    # GT Class filter
    if "gt_class_name" in df.columns:
        gt_classes = ["All"] + sorted(df["gt_class_name"].dropna().unique().tolist())
        selected_gt = st.sidebar.selectbox("GT Class", gt_classes)
    else:
        selected_gt = "All"
    # Pred Class filter
    if "pred_class_name" in df.columns:
        pred_classes = ["All"] + sorted(df["pred_class_name"].dropna().unique().tolist())
        selected_pred = st.sidebar.selectbox("Pred Class", pred_classes)
    else:
        selected_pred = "All"
    # Error Type filter
    if "error_type" in df.columns:
        error_types = ["All"] + sorted(df["error_type"].dropna().unique().tolist())
        selected_error = st.sidebar.selectbox("Error Type", error_types)
    else:
        selected_error = "All"
    # Confidence filter
    if "conf_score" in df.columns:
        min_conf, max_conf = float(df["conf_score"].min()), float(df["conf_score"].max())
        conf_range = st.sidebar.slider("Confidence range", min_value=min_conf, max_value=max_conf,
                                       value=(min_conf, max_conf), step=0.01)
    else:
        conf_range = (0.0, 1.0)

    # Apply filters
    filtered = df.copy()
    if selected_gt != "All": filtered = filtered[filtered["gt_class_name"] == selected_gt]
    if selected_pred != "All": filtered = filtered[filtered["pred_class_name"] == selected_pred]
    if selected_error != "All": filtered = filtered[filtered["error_type"] == selected_error]
    if "conf_score" in df.columns:
        filtered = filtered[(filtered["conf_score"] >= conf_range[0]) & (filtered["conf_score"] <= conf_range[1])]

    # ------------------------------
    # Sorting
    # ------------------------------
    sort_by = st.sidebar.selectbox("Sort by", ["precision", "recall", "f1", "conf_score", "gt_w", "gt_h"], index=0)
    sort_order = st.sidebar.radio("Order", ["Descending", "Ascending"], index=0)
    filtered = filtered.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))

    # ------------------------------
    # Stats Panel
    # ------------------------------
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

    # ------------------------------
    # Image Viewer
    # ------------------------------
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
                    # GT
                    if not pd.isna(row.get("gt_x_center")):
                        gt_x1 = row["gt_x_center"] * w - row["gt_w"] * w / 2
                        gt_y1 = row["gt_y_center"] * h - row["gt_h"] * h / 2
                        gt_x2 = row["gt_x_center"] * w + row["gt_w"] * w / 2
                        gt_y2 = row["gt_y_center"] * h + row["gt_h"] * h / 2
                        draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline="green", width=3)
                        draw.text((gt_x1, gt_y1 - 10), str(row["gt_class_name"]), fill="green")
                    # Pred
                    if not pd.isna(row.get("pred_x_center")):
                        pred_x1 = row["pred_x_center"] * w - row["pred_w"] * w / 2
                        pred_y1 = row["pred_y_center"] * h - row["pred_h"] * h / 2
                        pred_x2 = row["pred_x_center"] * w + row["pred_w"] * w / 2
                        pred_y2 = row["pred_y_center"] * h + row["pred_h"] * h / 2
                        color = ERROR_COLORS.get(row["error_type"], "yellow")
                        draw.rectangle([pred_x1, pred_y1, pred_x2, pred_y2], outline=color, width=3)
                        draw.text((pred_x1, pred_y1 - 10), f"{row['pred_class_name']} ({row['conf_score']:.2f})", fill=color)
                with cols[i % num_cols]:
                    st.image(img, caption=f"{image_id}", use_container_width=True)
                i += 1
                if i % num_cols == 0:
                    cols = st.columns(num_cols)

    # ------------------------------
    # Download filtered CSV
    # ------------------------------
    st.subheader("Download Filtered Results")
    st.download_button(
        label="Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv"
    )
