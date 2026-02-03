# Face Mask Detection using CNN

A deep learning project that uses Convolutional Neural Networks (CNN) to detect whether a person is wearing a face mask or not.

## Overview

This project implements a binary image classification model that can identify whether a person in an image is wearing a face mask. The model is built using TensorFlow/Keras and trained on the Face Mask Dataset from Kaggle.

## Dataset

**Source:** [Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

The dataset contains:
- 3,725 images of people wearing masks
- 3,828 images of people without masks
- Total: 7,553 images

## Requirements

```
tensorflow
keras
numpy
matplotlib
opencv-python
pillow
scikit-learn
kaggle
```

## Installation

1. Install required packages:
```bash
pip install tensorflow keras numpy matplotlib opencv-python pillow scikit-learn kaggle
```

2. Set up Kaggle API credentials:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Download and Prepare Dataset

```python
# Download dataset from Kaggle
!kaggle datasets download -d omkargurav/face-mask-dataset

# Extract the dataset
from zipfile import ZipFile
with ZipFile('face-mask-dataset.zip', 'r') as zip:
    zip.extractall()
```

### Training the Model

The model is trained on preprocessed images (128x128 RGB) with the following architecture:

- Conv2D layer (32 filters, 3x3 kernel)
- MaxPooling2D (2x2)
- Conv2D layer (64 filters, 3x3 kernel)
- MaxPooling2D (2x2)
- Flatten layer
- Dense layer (128 neurons, ReLU)
- Dropout (0.5)
- Dense layer (64 neurons, ReLU)
- Dropout (0.5)
- Output layer (2 neurons, Sigmoid)

Run the training script to train the model for 5 epochs.

### Making Predictions

```python
# Load and preprocess the input image
input_image_path = 'path/to/your/image.jpg'
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

# Predict
input_prediction = model.predict(input_image_reshaped)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 1:
    print('The person in the image is wearing a mask')
else:
    print('The person in the image is not wearing a mask')
```

## Model Performance

After training for 5 epochs:
- **Training Accuracy:** ~92%
- **Validation Accuracy:** ~93.5%
- **Test Accuracy:** ~92.7%
- **Test Loss:** ~0.19

## Model Architecture

| Layer Type | Output Shape | Parameters |
|------------|--------------|------------|
| Conv2D | (126, 126, 32) | 896 |
| MaxPooling2D | (63, 63, 32) | 0 |
| Conv2D | (61, 61, 64) | 18,496 |
| MaxPooling2D | (30, 30, 64) | 0 |
| Flatten | (57,600) | 0 |
| Dense | (128) | 7,372,928 |
| Dropout | (128) | 0 |
| Dense | (64) | 8,256 |
| Dropout | (64) | 0 |
| Dense | (2) | 130 |

## Data Preprocessing

1. Images are resized to 128x128 pixels
2. Converted to RGB format
3. Normalized by dividing pixel values by 255
4. Split into training (80%) and testing (20%) sets
5. Validation split of 10% from training data

## Labels

- **0:** No mask
- **1:** Wearing mask

## Features

- Binary image classification
- Real-time prediction capability
- High accuracy (~92.7%)
- Robust CNN architecture with dropout for regularization

## Future Improvements

- Increase number of training epochs
- Implement data augmentation
- Try transfer learning with pre-trained models (VGG16, ResNet)
- Deploy as a web application
- Add real-time video detection capability

## License

Dataset license: Unknown (refer to Kaggle dataset page)

## Acknowledgments

- Dataset provided by Omkar Gurav on Kaggle
- Built with TensorFlow and Keras
