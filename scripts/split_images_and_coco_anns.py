import numpy as np
import os
import json
import shutil

def move_images_and_split_coco(coco_json_path, dataset_root):
    """
    Moves images into train, validation, and test folders while also splitting the COCO JSON 
    annotations into corresponding train.json, val.json, and test.json files.

    Args:
        coco_json_path (str): Path to the COCO annotation JSON file.
        dataset_root (str): Path to the dataset root directory.

    Returns:
        None
    """

    # Paths
    raw_imgs_path = os.path.join(dataset_root, "images", "raw")
    train_dir = os.path.join(dataset_root, "images", "train")
    val_dir = os.path.join(dataset_root, "images", "val")
    test_dir = os.path.join(dataset_root, "images", "test")
    annotations_dir = os.path.join(dataset_root, "annotations")

    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Load image filenames and sort them
    files = sorted([f for f in os.listdir(raw_imgs_path) if f.endswith((".jpg", ".png", ".tiff"))])
    imgs = np.array(files)

    # Shuffle images with a fixed seed
    imgs = np.random.RandomState(seed=42).permutation(imgs)

    # Train-validation-test split
    train_idx = round(0.8 * len(imgs))
    val_idx = round(0.9 * len(imgs))

    train_files = set(imgs[:train_idx])
    val_files = set(imgs[train_idx:val_idx])
    test_files = set(imgs[val_idx:])

    # Move images to respective directories
    for filename in imgs:
        old_path = os.path.join(raw_imgs_path, filename)
        if filename in train_files:
            new_path = os.path.join(train_dir, filename)
        elif filename in val_files:
            new_path = os.path.join(val_dir, filename)
        else:
            new_path = os.path.join(test_dir, filename)

        shutil.copy(old_path, new_path)  # Use shutil for compatibility

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Helper function to filter images and annotations
    def filter_coco(images_subset):
        image_ids = {img["id"] for img in images_subset}
        return [anno for anno in coco_data["annotations"] if anno["image_id"] in image_ids]

    # Filter images and annotations for train/val/test
    train_images = [img for img in coco_data["images"] if img["file_name"] in train_files]
    val_images = [img for img in coco_data["images"] if img["file_name"] in val_files]
    test_images = [img for img in coco_data["images"] if img["file_name"] in test_files]

    train_annotations = filter_coco(train_images)
    val_annotations = filter_coco(val_images)
    test_annotations = filter_coco(test_images)

    # Create new COCO JSONs
    train_coco = {"images": train_images, "annotations": train_annotations, "categories": coco_data["categories"]}
    val_coco = {"images": val_images, "annotations": val_annotations, "categories": coco_data["categories"]}
    test_coco = {"images": test_images, "annotations": test_annotations, "categories": coco_data["categories"]}

    # Save new JSON files
    with open(os.path.join(annotations_dir, "train.json"), "w") as f:
        json.dump(train_coco, f, indent=4)

    with open(os.path.join(annotations_dir, "val.json"), "w") as f:
        json.dump(val_coco, f, indent=4)

    with open(os.path.join(annotations_dir, "test.json"), "w") as f:
        json.dump(test_coco, f, indent=4)

    print(f"✅ Dataset split completed! Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

def main():
    """
    Main function to split images and COCO annotations into train, validation, and test sets.
    """

    print(os.getcwd())
    move_images_and_split_coco(
        coco_json_path="./data/annotations/image-annotator_HIV_syphilis_coco.json",
        dataset_root="./data"
    )


if __name__ == "__main__":
    main()