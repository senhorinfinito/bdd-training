import os 
import os.path as osp
import json 
from utils.config_loader import logger, one_line_symbol
from utils.utils import yolo_writer
from PIL import Image
from tqdm import tqdm

class BDD:

    def __init__(self, config):
        self.det_train =  config.get("paths").get("det_train", None)
        self.det_val =  config.get("paths").get("det_val", None)
        self.img_dir = config.get("paths").get("image_root", None) 
        self.classes =  config.get("classes", [])
        self.label_dir  = config.get("paths").get("output_labels", "../dataset")
        os.makedirs(self.label_dir, exist_ok=True)
        
    def _get_images(self):
        _all_images = {}
        if self.img_dir is None:
            raise ValueError("Image directory path is not specified in the configuration.")
        
        for root, _, files in os.walk(self.img_dir):
            for filename in files:
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    _all_images[filename] = osp.join(root, filename)
        return _all_images

    def __get_img_info(self, img_path):
        try:
            with Image.open(img_path) as img:
                w, h =  img.size
            return w, h
        except FileNotFoundError:
            logger.error(f"Image not found: {img_path}")
        except OSError:
            logger.error(f"Error opening image: {img_path}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return None, None

    def __yolo_annot(self, classid, x1, x2, y1, y2, w, h):
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # normalize
        x_center /= w
        y_center /= h
        width /= w
        height /= h

        return classid, x_center, y_center, width, height

    def _get_labels(self, istrain=False, yolo=False):        
        all_labels = []
        __all_images = self._get_images()
        if self.det_train is None:
            raise ValueError("Detection training labels path is not specified in the configuration.")

        if istrain:
            try:
                logger.info(f"Loading train labels")

                with open(self.det_train, 'r') as f:
                    data = json.load(f)
            except Exception as e: 
                logger.error(f"Error loading training labels: {e}")
        else:
            logger.info(f"Loading validation labels")
            try:
                with open(self.det_val, 'r') as f:
                    data = json.load(f)
            except Exception as e: 
                logger.error(f"Error loading validation labels: {e}")

        for i in tqdm(range(len(data)), desc=f"Processing {'train' if istrain else 'val'} labels"):
            img_info =  data[i]
            __image_name  = img_info.get("name", "null")
            __weather =  img_info.get("attributes").get("weather", "null")
            __scene =  img_info.get("attributes").get("scene", "null")
            __labels = img_info.get("labels", [])
            w, h  = self.__get_img_info(__all_images[__image_name])
            yolo_annots = []
            for __label in __labels:
                category = __label.get("category", "null")
                if category not in self.classes:
                    continue
                __class_id = self.classes.index(category)
                __traffic_light_color = __label.get("attributes").get("trafficLightColor", "none")
                __is_occluded =  __label.get("attributes").get("occluded", "false")
                __is_truncated =  __label.get("attributes").get("truncated", "false")
                __bbox2d  = __label.get("box2d", {})
                x1, y1, x2, y2 = __bbox2d.get("x1", 0),__bbox2d.get("y1", 0),  __bbox2d.get("x2", 0), __bbox2d.get("y2", 0)
                class_id, x_center, y_center, width, height = self.__yolo_annot(__class_id, x1, x2, y1, y2, w, h )
                yolo_annots.append([class_id, x_center, y_center, width, height])
                all_labels.append({
                    "image_name": __image_name,
                    "width": w,
                    "height": h,
                    "weather": __weather,
                    "scene": __scene,
                    "category": category,
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_center": x_center,
                    "y_center": y_center,
                    "box_width": width,
                    "box_height": height,
                    "traffic_light_color": __traffic_light_color,
                    "is_occluded": __is_occluded,
                    "is_truncated": __is_truncated
                })

            if yolo:
                yolo_dir  = os.path.join(self.label_dir,  "labels")
                yolo_writer(yolo_annots, os.path.join(yolo_dir, "train" if istrain else "val", __image_name.replace(".jpg", ".txt")))
                del yolo_annots
                
        return all_labels