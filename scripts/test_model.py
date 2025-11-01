import torch
import torchvision
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import yaml


# Load config file
with open("configs/chembio_hiv_syph.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
model_date = config["model_date"]
model_path = f"{config['final_model_dir']}/{config['file_prefix']}_{config['model_name']}_final_{model_date}.pth"
test_img_dir = config["test_img_dir"]
test_ann_file = config["test_ann_file"]
class_names = config["class_names"]
num_classes = len(class_names) + 1  # NOTE Need to add the background class

# 🔹 Load trained model


def load_model(model_path, num_classes=4):
    if config['model_name'] == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT")
        # Update the box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        # And also for the mask predictor:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)

    elif config['model_name'] == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

    elif config['model_name'] == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None,  # no pretrained head
            weights_backbone="DEFAULT",  # pretrained backbone
            num_classes=num_classes
        )

    elif config['model_name'] == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=num_classes
        )

    model_path = f"{config['final_model_dir']}/{config['file_prefix']}_{config['model_name']}_final_{config['model_date']}.pth"
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device("cpu")))
    model.eval()

    return model

# 🔹 Load test dataset
def load_test_data(test_img_dir, test_ann_file):
    transform = T.Compose([T.ToTensor()])
    dataset = CocoDetection(
        root=test_img_dir, annFile=test_ann_file, transform=transform)
    return dataset

