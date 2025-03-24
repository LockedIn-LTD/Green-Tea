# Green-Tea: Drowsiness Detection Model

## Overview
Green Tea is a real-time drowsiness detection system that identifies signs of fatigue using a webcam. It analyzes eye openness and yawning behavior to detect drowsiness, helping improve driver safety and alertness.

## Features
- **Eye State Detection**: Detects left and right eye open/closed status.
- **Yawning Detection**: Uses a CNN-based PyTorch model to classify yawning.
- **Real-Time Processing**: Runs live via a webcam.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision opencv-python numpy dlib Pillow
```
Additionally, download `shape_predictor_68_face_landmarks.dat` for dlib's face landmark detection from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

## Usage
1. Clone the repository:
```bash
git clone https://github.com/LockedIn-LTD/Green-Tea.git
cd Green-Tea
```
2. Place `yawn_detection.pth` inside the `Yawning` directory.
3. Run the detection system:
```bash
python main.py
```

## Model Details
The system uses a CNN model trained on yawning images. The `yawn_detection.pth` model is required for yawning detection. It processes extracted mouth regions from video frames and classifies them as yawning or not.

## Training from Scratch
To train the yawning detection model from scratch, follow these steps:

### 1. Dataset Preparation
- Download the `yawn-dataset` from Kaggle.
- Extract the dataset and organize it into labeled folders (`Yawning`, `Not_Yawning`).

### 2. Data Preprocessing
- Resize images to 224x224 pixels.
- Convert images to PyTorch tensors.
- Normalize pixel values using mean=0.5 and std=0.5.

### 3. Model Architecture
The CNN consists of:
- **Three convolutional layers**: 32, 64, and 128 filters.
- **Max pooling layers** to reduce spatial dimensions.
- **Fully connected layers** for classification.
- **Activation function**: ReLU with softmax for the final output.

### 4. Training
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Train-Test Split: 80% training, 20% validation
- Hardware: Supports GPU acceleration (CUDA if available)

## Deployment
- **Windows**: The system is designed for Windows but is expected to work on Mac and ARM64 (e.g., Docker, Jetson Orin Nano).
- **Docker**: A Dockerfile can be created for easy deployment.

## License
This project is for educational purposes under LockedIn LTD. You may use an open-source license of your choice.

## Contact
For questions or contributions, please open an issue on the repository.

