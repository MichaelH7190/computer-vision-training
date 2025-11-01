from tqdm import tqdm
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDHead
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import numpy as np
import yaml

# Load config file
with open("configs/chembio_hiv_syph.yaml", "r") as f:
    config = yaml.safe_load(f)

# Model selection
model_name = config["model_name"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
save_interval = config["save_interval"]
file_prefix = config["file_prefix"]
class_names = config["class_names"]
num_classes = len(class_names) + 1  # NOTE: Need to add background class

# Paths
train_img_dir = config["train_img_dir"]
train_ann_file = config["train_ann_file"]
val_img_dir = config["val_img_dir"]
val_ann_file = config["val_ann_file"]
checkpoint_dir = config["checkpoint_dir"]
final_model_dir = config["final_model_dir"]
model_date = config["model_date"]

class CocoDatasetWithBoxes(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        boxes = []
        labels = []
        masks = []

        for annotation in target:
            bbox = annotation["bbox"]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(annotation["category_id"])
            masks.append(self.coco.annToMask(annotation))

        target_dict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
        }
        return img, target_dict

def get_transform():
    return T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    model.train()
    progress_bar = tqdm(
        data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if (batch_idx + 1) % 10 == 0:
            progress_bar.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(data_loader)
    print(f"✅ Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
    return avg_loss

def get_model(num_classes, model_name="maskrcnn"):
    if model_name == "maskrcnn":
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

    elif model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None,  # no pretrained head
            weights_backbone="DEFAULT",  # pretrained backbone
            num_classes=num_classes
)

    elif model_name == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=num_classes
    )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

if __name__ == "__main__":
    train_dataset = CocoDatasetWithBoxes(root="./data/images/train",
                                         annFile="./data/annotations/train.json", transform=get_transform())
    val_dataset = CocoDatasetWithBoxes(root="./data/images/val",
                                       annFile="./data/annotations/val.json", transform=get_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2,
                            shuffle=False, num_workers=2, collate_fn=collate_fn)

    # num_classes = 4  # 3 classes + 1 background
    # num_classes comes from config - not hardcoded above!
    model = get_model(num_classes, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    save_interval = 2
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_epochs)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/{file_prefix}_{model_name}_epoch_{epoch+1}_{model_date}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    final_model_path = f"{final_model_dir}/{file_prefix}_{model_name}_final_{model_date}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Model saved as {final_model_path}")
