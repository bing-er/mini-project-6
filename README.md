# Mini Project 6: Transfer Learning Application

**Course:** COMP 9130 — Applied Artificial Intelligence  
**Institution:** BCIT  
**Date:** February 25, 2026  

---

## Project Overview

This project demonstrates the practical application of **Transfer Learning** using the **Oxford 102 Flower Dataset**. A pre-trained **ResNet50** model (trained on ImageNet) is adapted to classify images into 102 flower categories.

Two transfer learning strategies are implemented and compared:

1. **Feature Extraction**
   - Freeze the ResNet50 backbone
   - Train only the custom classification head

2. **Fine-Tuning**
   - Unfreeze the last 50 layers of ResNet50
   - Continue training with a reduced learning rate

Additionally, **Grad-CAM visualization** is implemented to interpret model attention and highlight image regions that contribute most to predictions.

---

## Dataset
*   **Name:** Oxford 102 Flower Dataset
*   **Source:** [Kaggle - PyTorch Challenge Flower Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)
*   **Classes:** 102
*   **Training Set:** 6,552 images
*   **Validation Set:** 409 images
*   **Test Set:** 409 images

---

## Environment

This project is developed and executed locally using:

- Python 3.10+
- TensorFlow 2.16 (tensorflow-macos)
- tensorflow-metal (Apple Silicon GPU acceleration)
- Jupyter Notebook
- macOS (Apple M2 Max)

GPU availability can be verified using:

```python
import tensorflow as tf
tf.config.list_physical_devices("GPU")
```

---

## Methodology
### 1. Data Preparation
*   Images resized to `(224, 224)`.
*   Data augmentation applied: RandomFlip, RandomRotation, RandomZoom, RandomContrast.
*   Preprocessing: ResNet50 specific preprocessing.

### 2. Feature Extraction
*   **Base Model:** ResNet50 (ImageNet weights), all layers frozen.
*   **Head:** GlobalAveragePooling2D -> BatchNormalization -> Dense(512, ReLU) -> Dropout(0.5) -> Dense(102, Softmax).
*   **Optimizer:** Adam (lr=1e-3).
*   **Training:** Trained for up to 30 epochs with EarlyStopping.

### 3. Fine-Tuning
* Unfreeze the last 50 layers of the ResNet50 backbone
* Reduce learning rate to prevent catastrophic forgetting
* **Optimizer:** Adam (learning rate = 1e-5)
* **Callbacks:**
  * EarlyStopping
  * ReduceLROnPlateau


## Results
| Method | Test Accuracy | Test Loss  | F1 Score   | 
|---|---------------|------------|------------|
| **Feature Extraction** | 79.46%        | 1.8028     | 78.92%     |
| **Fine-Tuning** | **82.15%**    | **1.5789** | **81.66%** |

Fine-tuning improved test accuracy by approximately 11%, demonstrating the advantage of adapting deeper convolutional layers to the target dataset.

## 📁 Project Structure

```
mini-project-6/
├── data/                        
│   ├── test/
│   ├── train
│   ├── valid/
│   └── DATA_INSTRUCTIONS.txt
├── figures/                       # Output figures
├── models/                        # Final models only
├── notebooks/                          
│   └── transfer_learning_flowers102.ipynb    
├── src/
│   ├── utils.py
│   ├── compare.py
│   ├── gradcam.py
│   └── arch_compare.py
├── README.md                    # This file
├── requirements.txt             # Python dependencies
```


## 🚀 Running the Project (Local Setup)

1.  **Create Virtual Environment:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon GPU support:
```bash
pip install tensorflow-macos tensorflow-metal
```

2.  **Launch Jupyter Notebook**
```bash
jupyter notebook
```

Open:
```bash
notebooks/transfer_learning_flowers102.ipynb
```

Run all cells sequentially.

## Key Learning Outcomes

* Practical implementation of transfer learning
* Comparison between frozen backbone and fine-tuning strategies
* Learning rate sensitivity during deep model adaptation
* Model interpretability using Grad-CAM
* Efficient GPU training on Apple Silicon (Metal backend)

## Bonus
The project also includes **Grad-CAM** visualizations to interpret the model's predictions by highlighting the regions of the image that contributed most to the classification decision.

## References
Dataset: [Kaggle - PyTorch Challenge Flower Dataset](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset)

Model: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.