# Computer Vision Model Training Repository

This repository provides a modular framework for training, evaluating, and validating computer vision models using COCO-formatted datasets. The design is centered around the `core` branch, which serves as a stable foundation to branch off from and modify as needed for training new models.

---

## Repository Structure

```
.
├── README.md               # This documentation file
├── requirements.txt        # Python dependencies for the project
├── checkpoints             # Directory to store training checkpoints
├── configs                 # Directory to store training and testing configurations
├── models                  # Directory to store trained model weights
└── scripts                 # Contains training, testing, data splitting, and validation scripts
  ├── split_images_and_coco_anns.py  # Splits raw images and COCO annotations into train/val/test sets
  ├── test_model.py                  # Evaluates a trained model and computes mAP across IoU thresholds
  ├── train_model.py                 # Trains models (supports Mask R-CNN, Faster R-CNN, RetinaNet, SSD)
  └── validate_coco.py               # Validates COCO annotation files for correct formatting
```

Additionally, the data directory (not versioned in this repo) should follow this structure:

```
data/
├── annotations             # COCO JSON files for train, val, and test splits
├── images
│   ├── raw                 # Raw image files (input for splitting)
│   ├── train               # Training images (populated after splitting)
│   ├── val                 # Validation images (populated after splitting)
│   └── test                # Testing images (populated after splitting)
└── via_projects            # (Optional) VIA project files for annotation
```

## Configuration Files

This repository uses YAML configuration files to manage training parameters, dataset paths, and model storage settings. These configuration files allow users to modify settings without changing the core training or testing scripts.

### Default Configuration File

The default configuration file is located at:
`configs/default.yaml`

Example configuration:

```yaml
# General Training Parameters
model_name: "maskrcnn"
num_epochs: 10
batch_size: 2
learning_rate: 0.0001
save_interval: 2
file_prefix: "cassette_membrane_well"
num_classes: 4  # Number of classes (including background)

# Dataset Paths
train_img_dir: "./data/images/train"
train_ann_file: "./data/annotations/train.json"
val_img_dir: "./data/images/val"
val_ann_file: "./data/annotations/val.json"
test_img_dir: "./data/images/test"
test_ann_file: "./data/annotations/test.json"

# Model Saving Paths
checkpoint_dir: "./checkpoints"
final_model_dir: "./models"
model_date: "2021-09-01"
```

### How to Use Configuration Files

- Modify `configs/default.yaml` to customize training parameters.
- The `train_model.py` and `test_model.py` scripts will automatically read values from this file.
- You can create multiple configuration files for different experiments and specify which one to use by modifying your scripts.

### What Should Be Committed?

✅ Commit only configuration files (e.g., `configs/default.yaml`).

❌ Do not commit:
- Model outputs (`models/` directory)
- Checkpoints (`checkpoints/` directory)
- Large dataset files

To enforce this, `configs/` should remain part of `.gitignore`.

### Why This is Important?

- Keeps the repository lightweight by avoiding large model files.
- Ensures reproducibility by tracking only the essential configuration files.
- Allows multiple users to train different models without modifying the core code.

## Setup

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and Activate a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- **Organize Your Data:**
  Place your raw images in the `data/images/raw` folder and your original COCO JSON annotations in `data/annotations` (or an appropriate subfolder).

## Validating Annotations

- **Script:** `scripts/validate_coco.py`
- **Overview:**
  Ensures that the COCO JSON annotation files have correctly formatted bounding boxes. This is useful for catching errors in annotation files before training.

- **Usage:**
  ```bash
  python scripts/validate_coco.py
  ```

- **Splitting the Dataset:**
  Use the provided script to split images and annotations into training, validation, and test sets.
  ```bash
  python scripts/split_images_and_coco_anns.py
  ```
  This script:
  - Copies images from `data/images/raw` into `data/images/train`, `data/images/val`, and `data/images/test`
  - Splits the COCO annotations into corresponding `train.json`, `val.json`, and `test.json` files in the `data/annotations` folder

## Training a Model

- **Script:** `scripts/train_model.py`
- **Overview:**
  This script trains a computer vision model using COCO-formatted data. It supports multiple architectures (e.g., Mask R-CNN, Faster R-CNN, RetinaNet, SSD).

- **Steps:**
  1. **Configuration:**
   Modify parameters such as `model_name`, `num_epochs`, and file naming prefixes directly in the script.
  2. **Run Training:**
   Execute the script to start the training process.
   ```bash
   python scripts/train_model.py
   ```
  3. **Checkpoints & Final Model:**
   Checkpoints are saved periodically (default every 2 epochs) in a `checkpoints/` folder, and the final model is saved in the `models/` directory.

## Evaluating the Model

- **Script:** `scripts/test_model.py`
- **Overview:**
  This script loads a trained model, runs inference on a test dataset, and computes the mean Average Precision (mAP) over a range of IoU thresholds. It also generates a plot (`mAP_vs_IoU.png`) to visualize performance.

- **Usage:**
  Update the paths for the model weights, test images, and test annotations as needed, then run:
  ```bash
  python scripts/test_model.py
  ```

## Branching Strategy

- The `core` branch is intended as the main development branch from which you can branch off for experiments and modifications.
- When starting a new model training experiment or tweaking existing functionality, create a new branch from `core` to keep the main branch stable.
- Config files can be committed, but models, checkpoints, and data should not be committed to the repo. 
- Any updates should attempt to be backwards compatible, at least with work done in the last 6 months. Breaking changes should be clearly noted, and considered before being merged into `core`.

## Contributing

Contributions and improvements are welcome! Please open an issue or submit a pull request if you have suggestions or enhancements. Make sure to update the documentation accordingly.

## Acknowledgements

- This repository leverages the PyTorch and Torchvision libraries for model training and evaluation.
- Thanks to the open-source community for the tools and libraries that made this project possible.
