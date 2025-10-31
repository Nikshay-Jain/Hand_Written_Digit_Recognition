# 🔢 Handwritten Digit Recognition

A deep learning classifier for recognizing handwritten digits (0-9) from the MNIST dataset using TensorFlow/Keras, achieving ~97.5% test accuracy.

## ✨ Features

- **Deep Neural Network Architecture**: 3-layer fully-connected network optimized for digit classification
- **High Accuracy**: 97.96% training accuracy, 97.55% test accuracy on MNIST dataset
- **Model Optimization**: Experimental analysis of different hidden layer configurations
- **Visual Analysis**: Training performance visualization with matplotlib
- **Production-Ready**: Normalized preprocessing pipeline and efficient Adam optimizer

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow pandas numpy matplotlib
```

### Installation

```bash
git clone https://github.com/Nikshay-Jain/Hand_Written_Digit_Recognition.git
cd Hand_Written_Digit_Recognition
```

### Run the Model

Open and execute the Jupyter notebook:

```bash
jupyter notebook Digit-Recognizer.ipynb
```

## 🛠️ Usage

The notebook demonstrates a complete ML pipeline:

1. **Data Loading**: Automatic MNIST dataset import via TensorFlow
2. **Preprocessing**: Pixel normalization (0-255 → 0-1 range)
3. **Model Architecture**:
   - Input Layer: 784 features (28×28 pixels)
   - Hidden Layer 1: 144 units (ReLU)
   - Hidden Layer 2: 72 units (ReLU)
   - Output Layer: 10 units (Softmax)
4. **Training**: Adam optimizer with Sparse Categorical Crossentropy loss
5. **Evaluation**: Performance metrics and accuracy visualization

### Example Code Structure

```python
# Model is built and trained with:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(144, activation='relu'),
    tf.keras.layers.Dense(72, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## 📊 Model Performance

- **Training Accuracy**: 97.96%
- **Test Accuracy**: 97.55%
- **Dataset**: MNIST (60,000 training images, 10,000 test images)

## 🔧 Technology Stack

- **Framework**: TensorFlow/Keras
- **Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib
- **Environment**: Jupyter Notebook
