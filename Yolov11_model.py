"""
This CITATION.cff file was generated with https://bit.ly/cffinit

cff-version: 1.2.0
title: Ultralytics YOLO
message: >-
  If you use this software, please cite it using the
  metadata from this file.
type: software
authors:
  - given-names: Glenn
    family-names: Jocher
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0001-5950-6979'
  - family-names: Qiu
    given-names: Jing
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0003-3783-7069'
  - given-names: Ayush
    family-names: Chaurasia
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0002-7603-6750'
repository-code: 'https://github.com/ultralytics/ultralytics'
url: 'https://ultralytics.com'
license: AGPL-3.0
version: 8.0.0
date-released: '2023-01-10'



"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment_image(image, bbox, class_id):
    """
    Applies a set of augmentations to an image along with its bounding boxes.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be augmented.
    bbox : list of tuples
        A list of bounding boxes in YOLO format (x_center, y_center, width, height).
    class_id : list
        A list of class IDs corresponding to each bounding box.

    Returns:
    --------
    torch.Tensor
        The augmented image converted to a tensor.
    list of tuples
        The augmented bounding boxes in YOLO format.
    list
        The corresponding class IDs for the bounding boxes.

    Augmentations applied:
    ----------------------
    - Horizontal flip (50% chance)
    - Random brightness and contrast adjustment (50% chance)
    - Random shift, scale, and rotation (50% chance)
    - Gaussian blur (30% chance)
    - Conversion to PyTorch tensor
    """
    
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),  # Flips the image and bounding boxes horizontally
        A.RandomBrightnessContrast(p=0.5),  # Adjusts brightness and contrast randomly
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),  # Applies shift, scale, and rotation
        A.GaussianBlur(p=0.3),  # Applies Gaussian blur to reduce noise
        ToTensorV2()  # Converts image to PyTorch tensor format
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_id']))

    augmented = augmentations(image=image, bboxes=bbox, class_id=class_id)

    return augmented['image'], augmented['bboxes'], augmented['class_id']


def get_project_paths():
    """
    Returns a dictionary containing various important file paths for the project.

    Paths included:
    ---------------
    - "project": Root directory of the project.
    - "data_yaml": Path to the dataset configuration file (data.yaml).
    - "weights": Path to the YOLO model weights file (yolo11l.pt).
    - "output": Directory where training results are stored.
    - "test_images": Directory containing test images.

    Returns:
    --------
    dict
        A dictionary mapping descriptive keys to their corresponding Path objects.
    """

    project_path = Path("/home/d356/ARVR")
    return {
        "project": project_path,
        "data_yaml": project_path / "DATA/data.yaml",
        "weights": project_path / "MODEL/yolo11l.pt",
        "output": project_path / "runs/train/YOLOv11_Improved/2feb",
        "test_images": project_path / "DATA/test/images"
    }

def load_model(weights_path):
    """
    Loads a YOLO model from the specified weights file.

    Parameters:
    -----------
    weights_path : str or pathlib.Path
        Path to the YOLO model weights file.

    Returns:
    --------
    YOLO
        An instance of the YOLO model loaded with the given weights.
    """
    return YOLO(str(weights_path))


def train_model(model, data_yaml, output_dir):
    """
    Trains the YOLO model using the specified dataset and training parameters.

    Parameters:
    -----------
    model : YOLO
        The YOLO model instance to be trained.
    data_yaml : str or pathlib.Path
        Path to the dataset configuration file (data.yaml).
    output_dir : str or pathlib.Path
        Directory where training results will be saved.

    Returns:
    --------
    None
    """
    model.train(
        data=str(data_yaml),
        epochs=300,          # Number of training epochs
        batch=8,           # Batch size for training
        imgsz=640,         # Input image size
        optimizer='Adam',  # Optimization algorithm
        project=str(output_dir),
        name="YOLOv11_Improved",
    )


def evaluate_model(model):
    """
    Evaluates the trained YOLO model on a validation dataset.

    Parameters:
    -----------
    model : YOLO
        The trained YOLO model instance.

    Returns:
    --------
    dict
        A dictionary containing validation metrics such as precision, recall, and mAP.
    """
    metrics = model.val()
    print("Validation Results:", metrics)
    return metrics


def test_model(model, test_images_path):
    """
    Runs inference on test images using the trained YOLO model.

    Parameters:
    -----------
    model : YOLO
        The trained YOLO model instance.
    test_images_path : str or pathlib.Path
        Path to the directory containing test images.

    Returns:
    --------
    list
        A list of results containing predictions for each test image.
    """
    results = model.predict(
        source=str(test_images_path),
        save=True  # Saves the prediction results
    )
    print(f"Test Results saved to: {results}")
    return results


def main():
    """
    Main function to execute the YOLO model workflow.

    Workflow Steps:
    ---------------
    1. Load project paths.
    2. Load the pre-trained YOLO model using specified weights.
    3. Train the model on the dataset provided in `data.yaml`.
    4. Evaluate the trained model on validation data.
    5. Test the model on test images.

    Returns:
    --------
    None
    """
    paths = get_project_paths()  # Load project paths
    model = load_model(paths["weights"])  # Load YOLO model
    train_model(model, paths["data_yaml"], paths["output"])  # Train model
    evaluate_model(model)  # Evaluate model on validation data
    test_model(model, paths["test_images"])  # Test model on test images


if __name__ == "__main__":
    main()
