import os
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

# Dataset analysis 
from data.bdd_loader import BDD
from utils.config_loader import ConfigParser

# ------------------------------
# 1. Load config & dataset (cached)
# ------------------------------

@st.cache_data(show_spinner=False)  # disable default spinner
def load_datasets_with_progress():
    config = ConfigParser().get_data()
    yolo = bool(config.get("convert", None).get("yolo", False))
    bdd = BDD(config)

    progress = st.progress(0, text="Loading datasets...")
    
    # Step 1: load images
    imgs = bdd._get_images()
    progress.progress(20, text="Loaded image paths")

    # Step 2: load validation labels
    val_data = pd.DataFrame(bdd._get_labels(istrain=False, yolo=yolo))
    val_data["bbox_w"] = val_data["x2"] - val_data["x1"]
    val_data["bbox_h"] = val_data["y2"] - val_data["y1"]
    val_data["bboxpx"] = val_data["bbox_w"] * val_data["bbox_h"]
    progress.progress(50, text="Loaded validation annotations")

    # Step 3: load training labels
    train_data = pd.DataFrame(bdd._get_labels(istrain=True, yolo=yolo))
    train_data["bbox_w"] = train_data["x2"] - train_data["x1"]
    train_data["bbox_h"] = train_data["y2"] - train_data["y1"]
    train_data["bboxpx"] = train_data["bbox_w"] * train_data["bbox_h"]
    progress.progress(80, text="Loaded training annotations")

    # Done
    progress.progress(100, text="Dataset loaded successfully")

    return train_data, val_data, imgs


st.title("BBox Filter & Image Viewer")
with st.spinner("Preparing datasets..."):
    train_data, val_data, imgs = load_datasets_with_progress()

# ------------------------------
# 2. Streamlit UI setup
# ------------------------------
st.set_page_config(layout="wide")
# ------------------------------
# Extra: Train vs Val comparison
# ------------------------------

def get_category_stats(df):
    """Return per-category bbox_count and image_count."""
    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "bbox_count"]

    img_counts = df.groupby("category")["image_name"].nunique().reset_index()
    img_counts.columns = ["category", "image_count"]

    return pd.merge(cat_counts, img_counts, on="category")

# compute stats
train_stats = get_category_stats(train_data) if "category" in train_data.columns else pd.DataFrame()
val_stats = get_category_stats(val_data) if "category" in val_data.columns else pd.DataFrame()

if not train_stats.empty and not val_stats.empty:
    st.subheader("Train vs Val Category Comparison")

    # Merge train + val stats
    merged_stats = pd.merge(
        train_stats, val_stats, on="category", how="outer", suffixes=("_train", "_val")
    ).fillna(0)

    # Show comparison table
    st.dataframe(merged_stats, use_container_width=True, height=400)

    # Bar chart using Altair
    import altair as alt
    chart_data = merged_stats.melt(
        id_vars="category",
        value_vars=["image_count_train", "image_count_val"],
        var_name="dataset",
        value_name="count"
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title="Category"),
            y=alt.Y("count:Q", title="Image Count"),
            color="dataset:N",
            tooltip=["category", "dataset", "count"]
        )
        .properties(width=500, height=400)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Category comparison unavailable (no category column in one of the datasets).")


# Dataset selector
dataset_choice = st.radio("Select dataset", ["train", "val"], horizontal=True)
if dataset_choice == "train":
    data = train_data.copy()
else:
    data = val_data.copy()

# threshold slider
threshold = st.slider(
    "Select bbox threshold (px)",
    min_value=16,
    max_value=100,
    value=16,
    step=1
)

# ------------------------------
# 3. Filters in one row
# ------------------------------
filters = {}
cols = st.columns(5)

if "scene" in data.columns:
    scene_options = ["All"] + sorted(data["scene"].unique().tolist())
    filters["scene"] = cols[0].selectbox("Scene", scene_options)

if "category" in data.columns:
    cat_options = ["All"] + sorted(data["category"].unique().tolist())
    filters["category"] = cols[1].selectbox("Category", cat_options)

if "is_occluded" in data.columns:
    occ_options = ["All"] + sorted(data["is_occluded"].unique().tolist())
    filters["is_occluded"] = cols[2].selectbox("Occluded", occ_options)

if "is_truncated" in data.columns:
    trunc_options = ["All"] + sorted(data["is_truncated"].unique().tolist())
    filters["is_truncated"] = cols[3].selectbox("Truncated", trunc_options)

if "weather" in data.columns:
    weather_options = ["All"] + sorted(data["weather"].unique().tolist())
    filters["weather"] = cols[4].selectbox("Weather", weather_options)

# ------------------------------
# 4. Apply filters
# ------------------------------
mask = (data["bbox_w"] < threshold) & (data["bbox_h"] < threshold)
filtered = data.loc[mask]

for col, val in filters.items():
    if val != "All":
        filtered = filtered[filtered[col] == val]

st.write(f"Total boxes under threshold {threshold}px after filters: {len(filtered)}")

# ------------------------------
# 5. Draw bboxes
# ------------------------------
def draw_bboxes(image_path, df_subset):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, row in df_subset.iterrows():
        box = [row["x1"], row["y1"], row["x2"], row["y2"]]
        draw.rectangle(box, outline="red", width=3)
        if "category" in row:
            draw.text((row["x1"], row["y1"] - 10), str(row["category"]), fill="red")
    return img

# ------------------------------
# 6. Layout: left images | right stats
# ------------------------------
left_col, right_col = st.columns([3, 1], gap="large")

# Make right column fixed (sticky) using CSS
st.markdown(
    """
    <style>
    /* Right column sticky */
    [data-testid="column"]:nth-of-type(2) {
        position: sticky;
        top: 70px;   /* adjust offset for header height */
        align-self: flex-start;
        height: fit-content;
    }
    /* Left column should scroll */
    [data-testid="column"]:nth-of-type(1) {
        overflow-y: auto;
        max-height: 85vh; /* scrollable only for left side */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Right column → stats
with right_col:
    if "category" in data.columns and not filtered.empty:
        cat_counts = filtered["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "bbox_count"]

        img_counts = filtered.groupby("category")["image_name"].nunique().reset_index()
        img_counts.columns = ["category", "image_count"]

        cat_stats = pd.merge(cat_counts, img_counts, on="category")

        st.subheader("Category Stats")
        st.dataframe(cat_stats, use_container_width=True, height=300)

        import altair as alt
        chart_data = cat_stats.melt(
            id_vars="category",
            value_vars=["bbox_count", "image_count"],
            var_name="metric",
            value_name="count"
        )
        max_val = chart_data["count"].max()
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("category:N", title="Category"),
                y=alt.Y("count:Q", scale=alt.Scale(domain=[0, max_val]), title="Count"),
                color="metric:N",
                tooltip=["category", "metric", "count"]
            )
            .properties(width=300, height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No `category` column found or no filtered data.")

# Left column → images
with left_col:
    if not filtered.empty:
        grouped = filtered.groupby("image_name")
        cols = st.columns(4)  # 4 images per row
        i = 0
        for image_name, group in grouped:
            if image_name in imgs:
                image_path = imgs[image_name]
                if os.path.exists(image_path):
                    img = draw_bboxes(image_path, group)
                    with cols[i % 4]:
                        st.image(img, caption=image_name, use_container_width=True)
                i += 1
    else:
        st.info("No bounding boxes match the selected filters.")
