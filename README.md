# FCOS_Implementation

This repository contains a TensorFlow 2 implementation of the **FCOS (Fully Convolutional One-Stage Object Detection)** model, as described in the paper [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355).

The primary goal of this project was to provide an implementation for the [TensorFlow Models repository](https://github.com/tensorflow/models) to address [issue #10275](https://github.com/tensorflow/models/issues/10275).

## Blog Post

A detailed explanation of the implementation process, the challenges faced, and the architectural details of the model can be found in the author's Medium blog post:

**[Deep Learning model research implementation: FCOS](https://medium.com/@samiratra95/deep-learning-model-research-implementation-fcos-cc16507088c9)**

## Project Status

**On Hold:** This project is currently on hold due to a lack of sufficient computational resources required for training the model on the COCO dataset. The model has been implemented and compiles successfully, but has not been trained.

## Background

FCOS is a fully convolutional one-stage object detector that is anchor-box free and proposal free. It solves object detection in a per-pixel prediction fashion, similar to semantic segmentation. This approach avoids the complicated computation related to anchor boxes and the associated hyperparameters, making the detector much simpler and more flexible.

## Features

*   **Anchor-Free:** No need for anchor boxes, which simplifies the training process and reduces the number of hyperparameters.
*   **Fully Convolutional:** The entire network is fully convolutional, making it simple and flexible.
*   **FPN for Multi-level Prediction:** Uses a Feature Pyramid Network (FPN) to detect objects at different scales.
*   **Center-ness Branch:** A dedicated branch to predict the "center-ness" of a location, which helps to suppress low-quality detected bounding boxes.
*   **Built with TensorFlow 2 and Keras:** The model is implemented using the high-level Keras API in TensorFlow 2.

## Model Architecture

The model is composed of three main parts:

1.  **Backbone:** A pre-trained **ResNet-50** network on ImageNet is used as the backbone for feature extraction. The implementation takes the feature maps C3, C4, and C5 from the backbone.
2.  **Feature Pyramid Network (FPN):** An FPN is built on top of the backbone to generate a rich, multi-scale feature pyramid (P3, P4, P5, P6, P7).
3.  **Heads:** Three parallel prediction heads are attached to each level of the FPN:
    *   **Classification Head:** Predicts the class of the object at each location.
    *   **Regression Head:** Predicts the bounding box coordinates (left, top, right, bottom distances).
    *   **Center-ness Head:** Predicts the center-ness of each location.

## Dataset

This implementation is designed to be trained on the **COCO (Common Objects in Context) 2014 dataset**.

### Download

You can download the dataset using the following commands:

```bash
# Download training, validation, and test images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Unzip the files
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
unzip annotations_trainval2014.zip
```

The data loading script (`src/Data/load_data.py`) expects the data to be in a directory structure as defined by the paths in `main.py`. You will need to update these paths to match your local setup.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Samir-atra/FCOS_Implementation.git
    cd FCOS_Implementation
    ```

2.  **Set up the Python environment:**
    It is recommended to use a virtual environment.

3.  **Install dependencies:**
    The required packages are listed in `main.py`. You can install them using pip:
    ```bash
    pip install tensorflow numpy opencv-python
    ```
    The author also notes that some official TensorFlow models might be required, which can be installed by cloning the `tensorflow/models` repository.

## How to Run

1.  **Set the `PYTHONPATH`:**
    To ensure that the Python modules in the `src` directory are importable, add the project directory to your `PYTHONPATH`.
    ```bash
    export PYTHONPATH=$PYTHONPATH:`pwd`
    ```

2.  **Update file paths:**
    In `main.py` and `src/Data/load_data.py`, update the paths to the COCO dataset to match your local directory structure.

3.  **Run the training script:**
    ```bash
    python main.py
    ```
    Note that the script is currently set up to load a small subset of the data due to resource limitations.

## Implementation Details

*   **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of 0.01, momentum of 0.9, and weight decay of 0.0001.
*   **Learning Rate Schedule:** The learning rate is divided by 10 at 60,000 and 80,000 iterations.
*   **Loss Functions:**
    *   **Classification:** Focal Loss (`CategoricalFocalCrossentropy`) with alpha=0.25 and gamma=2.0.
    *   **Regression:** IoU Loss (custom implementation in `src/loss.py`).
    *   **Center-ness:** Binary Cross-Entropy (`BinaryCrossentropy`).

## Future Work

The author has identified several potential improvements for this implementation:

*   Use **ResNeXt** as the backbone for better performance.
*   Implement **GIoU (Generalized IoU) loss** instead of IoU loss for the regression task.
*   Add **Group Normalization** to the prediction heads.
*   Train the model on the full COCO dataset with sufficient computational resources.

## File Descriptions

*   `main.py`: The main entry point for training the model.
*   `script.sh`: Script to set the `PYTHONPATH`.
*   `Notes.txt`: The author's personal notes and development log.
*   `src/Data/load_data.py`: Script for loading and preprocessing the COCO dataset.
*   `src/Data/downloading_COCO.sh`: Commands to download the COCO dataset.
*   `src/model/model.py`: Defines the main FCOS model class and compiles it.
*   `src/model/backbone.py`: Defines the ResNet-50 backbone.
*   `src/model/pyramid.py`: Implements the Feature Pyramid Network (FPN).
*   `src/model/heads.py`: Implements the prediction heads.
*   `src/loss.py`: Contains the custom IoU loss implementation.
*   `Literature/`: A collection of relevant research papers.

## References

*   [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)
*   [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
*   [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)
*   [Keras RetinaNet Implementation Example](https://keras.io/examples/vision/retinanet/)

## License

This project is licensed under the terms of the **LICENSE** file.
