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

# 🔹 Compute mAP for different IoU thresholds (overall and per class)
def compute_map_vs_iou(model, dataset, device, class_names, iou_thresholds=np.arange(0.3, 0.95, 0.05), score_threshold=0.5):
    model.to(device)
    model.eval()

    overall_map = {}
    class_map = {class_name: [] for class_name in class_names}

    for iou_threshold in iou_thresholds:
        all_true_boxes = {cls: [] for cls in class_names}
        all_pred_boxes = {cls: [] for cls in class_names}
        all_pred_scores = {cls: [] for cls in class_names}

        for img, target in dataset:
            img_tensor = img.to(device).unsqueeze(0)

            with torch.no_grad():
                prediction = model(img_tensor)

            pred_boxes = prediction[0]["boxes"].cpu().numpy()
            pred_scores = prediction[0]["scores"].cpu().numpy()
            pred_labels = prediction[0]["labels"].cpu().numpy()

            true_boxes = np.array([obj["bbox"] for obj in target])
            true_labels = np.array([obj["category_id"] for obj in target])

            # Convert [x, y, w, h] → [x_min, y_min, x_max, y_max]
            if len(true_boxes) > 0:
                true_boxes[:, 2:] += true_boxes[:, :2]

            # Filter predictions by score threshold
            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            # Store per-class results
            # Start at 1 (COCO class IDs)
            for cls_id, cls_name in enumerate(class_names, start=1):
                cls_true_boxes = true_boxes[true_labels == cls_id]
                cls_pred_boxes = pred_boxes[pred_labels == cls_id]
                cls_pred_scores = pred_scores[pred_labels == cls_id]

                all_true_boxes[cls_name].append(cls_true_boxes)
                all_pred_boxes[cls_name].append(cls_pred_boxes)
                all_pred_scores[cls_name].append(cls_pred_scores)

        # Compute mAP for each class
        for cls_name in class_names:
            class_ap = compute_ap(
                all_true_boxes[cls_name], all_pred_boxes[cls_name], all_pred_scores[cls_name], iou_threshold)
            class_map[cls_name].append(class_ap)

        # Compute overall mAP (average over classes)
        overall_map[iou_threshold] = np.mean(
            [class_map[cls_name][-1] for cls_name in class_names])

    return overall_map, class_map


def compute_ap(all_true_boxes, all_pred_boxes, all_pred_scores, iou_threshold):
    """Helper function to compute Average Precision (AP) at a given IoU threshold."""
    tp, fp, total_gt = 0, 0, 0

    for true_boxes, pred_boxes, pred_scores in zip(all_true_boxes, all_pred_boxes, all_pred_scores):
        if len(true_boxes) == 0:
            fp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            total_gt += len(true_boxes)
            continue

        ious = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes))
        max_ious, _ = ious.max(dim=1)

        tp += (max_ious >= iou_threshold).sum().item()
        fp += (max_ious < iou_threshold).sum().item()
        total_gt += len(true_boxes)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (total_gt + 1e-6)
    return precision * recall


# 🔹 Plot overall and per-class mAP vs. IoU threshold
def plot_map_vs_iou(overall_map, class_map, iou_thresholds):
    plt.figure(figsize=(10, 6))

    # Plot overall mAP
    plt.plot(iou_thresholds, list(overall_map.values()),
             label="Overall mAP", marker="o", linestyle="-", color="b")

    # Plot per-class mAP
    for class_name, mAP_values in class_map.items():
        plt.plot(iou_thresholds, mAP_values,
                 label=f"{class_name} mAP", marker="o", linestyle="--")

    plt.xlabel("IoU Threshold")
    plt.ylabel("mAP")
    plt.title("Overall and Class-wise mAP vs. IoU Threshold")
    plt.legend()
    plt.grid()
    plt.savefig(f"mAP_vs_IoU_{config['file_prefix']}_{config['model_name']}_final_{model_date}.png")
    plt.show()


# 🔹 Run evaluation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    # model_path = "./models/cassette_membrane_well_final.pth"
    test_img_dir = "./data/images/test/"
    test_ann_file = "./data/annotations/test.json"

    # Class names (from your dataset)
    # TODO this should come from config also
    # class_names = ["casette", "membrane", "buffer", "buffer_sample"]

    # Load model and dataset
    print('loading model')
    model = load_model(model_path, num_classes).to(device)
    print('loading dataset')
    dataset = load_test_data(test_img_dir, test_ann_file)

    # Compute mAP for different IoU thresholds
    print('computing mAP')
    iou_thresholds = np.arange(0.3, 0.95, 0.05)
    overall_map, class_map = compute_map_vs_iou(
        model, dataset, device, class_names, iou_thresholds)

    # Plot mAP vs. IoU threshold
    plot_map_vs_iou(overall_map, class_map, iou_thresholds)
