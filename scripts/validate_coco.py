import json
import os

def validate_coco_annotations(coco_path):
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
    
    print(f"Validating {coco_path}...")
    for annotation in coco_data["annotations"]:
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            print(f"Invalid bbox in annotation ID {annotation['id']}: {bbox}")


if __name__ == "__main__":
    annotations_path = "./data/annotations"
    validate_coco_annotations(os.path.join(annotations_path, "train.json"))
    validate_coco_annotations(os.path.join(annotations_path, "val.json"))
    validate_coco_annotations(os.path.join(annotations_path, "test.json"))
