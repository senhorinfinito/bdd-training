# Dataset analysis
import os
from pandas import DataFrame
from data.bdd_loader import BDD
from utils.config_loader import ConfigParser
from data.stats import VisDataset
from utils.utils import convert_to_yolo, filter_by_min_size

config = ConfigParser().get_data()
yolo = bool(config.get("convert", None).get("yolo", True))
bdd = BDD(config)
imgs = bdd._get_images()
train_data = bdd._get_labels(
    istrain=True, yolo=yolo
)  # check unique labels in the dataset
val_data = bdd._get_labels(
    istrain=False, yolo=yolo
)  # check unique labels in the dataset

vis = VisDataset(config, train_data, val_data)
plot_data = vis.plot_all(istrain=True)
plot_data = vis.plot_all(istrain=False)
plot_compare = vis.compare_all()

#  create trainable data after removing noising dataset.

#  removing the annotations with with lower resolution for particular objects
#  person - 16px
#  Car, bus, train, truck,  motor - 32px

filter_criteria = {
    "person": 16,
    "bus": 32,
    "car": 32,
    "train": 32,
    "truck": 32,
    "motor": 32,
}


train_df = DataFrame(train_data)
val_df = DataFrame(val_data)

train_filtered = filter_by_min_size(train_df, filter_criteria)
val_filtered = filter_by_min_size(val_df, filter_criteria)

_filtered = VisDataset(config, train_filtered, val_filtered, isfiltered=True)
plot_data = _filtered.plot_all(istrain=True)
plot_data = _filtered.plot_all(istrain=False)
plot_compare = _filtered.compare_all()

filter_train_labels = os.path.join(
    config.get("paths", {}).get("output_labels", "../dataset"), "labels_filtered"
)
filter_val_labels = os.path.join(
    config.get("paths", {}).get("output_labels", "../dataset"), "labels_filtered"
)
os.makedirs(filter_train_labels, exist_ok=True)
os.makedirs(filter_val_labels, exist_ok=True)

convert_to_yolo(train_filtered, imgs, filter_train_labels, "train")
convert_to_yolo(val_filtered, imgs, filter_val_labels, "val")
