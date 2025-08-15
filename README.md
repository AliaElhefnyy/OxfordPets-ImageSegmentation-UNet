# 🐾 Oxford Pets Image Segmentation with U-Net

This repository contains an implementation of a **U-Net** model for semantic segmentation of the **Oxford-IIIT Pet Dataset**, predicting pixel-level masks for pet images. The goal is to accurately segment each pet from the background, enabling applications in computer vision such as animal recognition, background removal, and automated annotation.

---

## 📂 Dataset
We use the **Oxford-IIIT Pet Dataset** from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet).  
It contains:
- **37 classes** (different breeds)
- RGB pet images of varying resolutions
- Pixel-wise segmentation masks (`1` = pet, `2` = outline, `0` = background)

---

## 🛠 Preprocessing & Data Pipeline

To ensure efficient and accurate training, we apply the following preprocessing steps:

1. **Image Resizing**: All images and masks are resized to **128×128** pixels.  
2. **Normalization**: Images scaled to `[0,1]` range; masks kept as integer labels for correct class mapping.  
3. **Data Augmentation**:
   - Random horizontal and vertical flips
   - Random rotations
4. **`tf.data` Optimizations**:
   - `cache()` for faster repeated access
   - `shuffle(1000)` for randomness in training batches
   - `batch(64)` for parallel GPU processing
   - `prefetch(tf.data.AUTOTUNE)` to overlap data loading with model execution

---

## 🧠 Model Architecture
The model is based on **U-Net**, a convolutional neural network widely used for segmentation tasks. Key features:
- **Encoder-Decoder** design
- **Skip connections** to preserve spatial details
- Output layer with `softmax` activation for multi-class segmentation

---

## 🎯 Training Strategy
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Optimizer**: Adam (`lr=0.001`)
- **Metrics**: Accuracy, Dice Coefficient, Mean IoU
- **Callbacks**:
  - **EarlyStopping**: Stops training if validation loss does not improve for 10 epochs
  - **ModelCheckpoint**: Saves best weights based on lowest validation loss
  - **ReduceLROnPlateau**: Lowers learning rate when progress plateaus

---

## 📊 Results & Insights

Training demonstrated consistent improvement across accuracy, Dice coefficient, and mean IoU, with rapid convergence in the early epochs.

**Best epoch (23) performance**:
- **Training Accuracy**: 93.69%
- **Training Dice Coefficient**: 0.6285
- **Training Mean IoU**: 0.8058
- **Validation Accuracy**: 91.70%
- **Validation Dice Coefficient**: 0.6139
- **Validation Mean IoU**: 0.7708
- **Validation Loss**: 0.2408


These results show that the U-Net learned robust segmentation boundaries and generalized well to unseen data.

---

## 📷 Example Predictions

Below are example predictions comparing:
1. Original Image
2. Ground Truth Mask
3. Predicted Mask

<img width="831" height="577" alt="Screenshot 2025-08-15 205116" src="https://github.com/user-attachments/assets/b1478d02-c6cd-4c17-9edf-1b34673c9172" />


---
## Install dependencies with

- pip install -r requirements.txt

---





