# 🐶🐱 VGG16 Transfer Learning - Cat vs Dog Classification

This project applies **transfer learning** using a pre-trained **VGG16** model to classify images of **cats** and **dogs**. The model is fine-tuned for binary classification on a custom dataset from Kaggle.

## 📂 Dataset

- Source: [Cats vs Dogs Classification Dataset](https://www.kaggle.com/datasets/vrajesh0sharma7/cat-vs-dog-classification/data)
- Structure:
  ```
  dogs_vs_cats/
    ├── train/
    └── test/
  ```

- 2 classes: `cats`, `dogs`

## 🚀 Approach

- Model: Pretrained `VGG16` from PyTorch's `torchvision.models`, with the classifier head replaced.
- Only the final classifier layers are trained; feature extractor is frozen.
- Input images resized to **160x160**.
- Training for **15 epochs** with **RMSprop** optimizer and **BCEWithLogitsLoss**.
- Data augmentation applied to the training set.

## 📊 Model Architecture

```text
VGG16 (features frozen) ➝ AdaptiveAvgPool2d ➝ Flatten ➝ Dropout ➝ Linear(512→1)
```

Only **513 trainable parameters** (final layer), rest of the model (14M+ params) is frozen.

## 🧪 Training Results

| Metric             | Final Value |
|--------------------|-------------|
| Training Accuracy  | 92.87%      |
| Test Accuracy      | **96.00%**  |
| Training Loss      | 0.2331      |
| Test Loss          | 0.1535      |

Training and validation metrics are plotted over epochs to visualize performance improvement.

## 📦 Dependencies

- Python 3.10+
- PyTorch, Torchvision
- Matplotlib, NumPy, tqdm
- Jupyter / Kaggle Notebook environment

## 📎 Links

- 📁 [Kaggle Notebook (Source Code)](https://www.kaggle.com/code/khoibuivn/vgg16-transfer-learning-cag-dog-classification)
- 🐱 [Dataset on Kaggle](https://www.kaggle.com/datasets/vrajesh0sharma7/cat-vs-dog-classification/data)

## 🛠️ How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib tqdm

# (Optional) Run notebook or script
# Jupyter / Kaggle environment recommended
```

## 📌 Key Highlights

- Used **transfer learning** for faster convergence and higher accuracy.
- **Data augmentation** improves generalization (random flip, rotation).
- Achieved **96% accuracy** on the test set with minimal training time.
