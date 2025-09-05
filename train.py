from roboflow import Roboflow
from torch.utils.data import Dataset
import json
from PIL import Image
import supervision as sv
import numpy as np
from collections import defaultdict


########## dataset ##########
def download_dataset(api_key, workspace, project_name, version_num):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("coco")
    return dataset

class RoboflowDataset(Dataset):
    def __init__(self, dataset_path, split):
        self.split = split

        sv_dataset = sv.DetectionDataset.from_coco(
            f"{dataset_path}/{split}/",
            f"{dataset_path}/{split}/_annotations.coco.json"
        )

        self.dataset = sv_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        filename, image, ann = self.dataset[idx] #ann.__class__ :supervision.detection.core.Detections        
        target = ann.xyxy
        class_id = ann.class_id
        bbox_area = ann.area
        
        # CLASSES = list(range(10))
        # label_counts = {}
        # for class_idx, label in enumerate(CLASSES):
        #   count = len(ann[ann.class_id == (class_idx)]) # Counts the number of annotations with that class
        #   if count == 0:
        #     continue
        #   label_counts[label] = count

        class_id_to_bbox = defaultdict(list)
        for obj_class, bbox in zip(class_id, target):
            class_id_to_bbox[int(obj_class)-1].append(bbox.tolist())

        return filename, Image.fromarray(image), dict(class_id_to_bbox)

# ds = download_dataset("4BDHggHM6vkVOoK3g0s3", "objectdetvlm", "water-meter-jbktv-7vz5k-fsod-ftoz-qii9s", 1)
# datasets = {
#     "train": RoboflowDataset(ds.location,"train"),
#     "val": RoboflowDataset(ds.location,"valid"),
#     "test": RoboflowDataset(ds.location,"test"),
# }
# datasets['train'].__getitem__(0)


########## model #######

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained(
#     "vikhyatk/moondream2",
#     revision="2025-06-21",
#     trust_remote_code=True,
#     device_map={"": "cuda"}
# )

# def denorm_pixel(box, width, height):
#     x_min = min(int(box['x_min'] * width), width - 1)
#     x_max = min(int(box['x_max'] * width), width - 1)
#     y_min = min(int(box['y_min'] * height), height - 1)
#     y_max = min(int(box['y_max'] * height), height - 1)

#     x_min = max(0, x_min); y_min = max(0, y_min)
#     if x_max < x_min: x_max = x_min
#     if y_max < y_min: y_max = y_min
#     return {'x_min_px': x_min, 'y_min_px': y_min,
#             'x_max_px': x_max, 'y_max_px': y_max}


# ds = datasets["test"]
# for image_idx in range(len(ds)):
# # image_idx = random.randint(0, len(ds))
#     image, ann = ds.__getitem__(image_idx)
#     for class_id, bbox_list in ann.items():
#         out = model.detect(image, "The digit " + str(class_id))
#         preds = [list(_.values()) for _ in out['objects']]
        
#         #metric  

        
#         # coco_bbox_list = []
#         # for bbox in bbox_list:
#             # x_min, y_min, x_max, y_max = bbox
#             # bbox_w = x_max - x_min
#             # bbox_h = y_max - y_min
#             # coco_bbox_list.append([x_min, y_min, bbox_w, bbox_h])
#         # plot_bbox(image,coco_bbox_list, class_id)
        
#     print("--"*100)

# image = Image.open(image_path)
# w,h = image.size
# for inp in ["0", "second 0"]:
#     out = model.detect(image, inp)
#     for pred in out['objects']:
#         x_min, y_min, x_max, y_max = list(denorm_pixel(pred, w, h).values())
#         bbox_w = x_max - x_min
#         bbox_h = y_max - y_min
#         plot_bbox(image, [x_min, y_min, bbox_w, bbox_h], inp)