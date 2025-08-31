import os
import shutil
from tqdm import tqdm
from pandas import DataFrame


def yolo_writer(annots, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for annot in annots:
            f.write(" ".join(str(x) for x in annot) + "\n")


def convert_to_yolo(df, imgs, output_root, split):
    """
    Convert dataframe annotations to YOLO format and copy images.

    Args:
        df (pd.DataFrame): annotations dataframe with columns
                           [image_name, x1, y1, x2, y2, width, height, class_id].
        imgs (dict): mapping {image_name: image_path}.
        output_root (str): main output folder, e.g., "output_labels".
        split (str): "train" or "val".
    """
    label_dir = os.path.join(output_root, split, "labels")
    image_dir = os.path.join(output_root, split, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    for image_name, group in tqdm(
        df.groupby("image_name"), desc=f"Converting {split} to YOLO format"
    ):
        annots = []
        for _, row in group.iterrows():
            annots.append(
                [
                    int(row["class_id"]),
                    round(row["x_center"], 6),
                    round(row["y_center"], 6),
                    round(row["box_width"], 6),
                    round(row["box_height"], 6),
                ]
            )

        # Save YOLO labels
        txt_file = os.path.join(label_dir, f"{os.path.splitext(image_name)[0]}.txt")
        yolo_writer(annots, txt_file)

        # Copy image to YOLO images folder
        if image_name in imgs:
            src_img = imgs[image_name]
            if os.path.exists(src_img):
                dst_img = os.path.join(image_dir, image_name)
                shutil.copy(src_img, dst_img)


def filter_by_min_size(
    df: DataFrame, criteria: dict, drop_empty_images: bool = True
) -> DataFrame:
    """
    Filters out rows where bounding boxes are smaller than thresholds in `criteria`.
    Optionally removes entire images if all their annotations are dropped.
    """

    def row_ok(row):
        min_size = criteria.get(row["category"], None)
        if min_size is None:
            return True  # keep if class not in criteria

        box_width = row["x2"] - row["x1"]
        box_height = row["y2"] - row["y1"]

        return box_width >= min_size and box_height >= min_size

    # Filter rows
    filtered = df[df.apply(row_ok, axis=1)].reset_index(drop=True)

    if drop_empty_images:
        # Drop images that no longer have any valid annotations
        valid_images = set(filtered["image_name"].unique())
        filtered = filtered[filtered["image_name"].isin(valid_images)].reset_index(
            drop=True
        )

    return filtered
