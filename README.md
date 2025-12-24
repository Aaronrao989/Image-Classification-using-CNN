<div align="center">

# 🔮 CIFAR-10 Image Classifier

### Deep Learning Powered Image Recognition

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://image-classification-using--cnn.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <img src="https://img.shields.io/badge/accuracy-85%25+-brightgreen?style=flat-square" alt="Accuracy">
  <img src="https://img.shields.io/badge/model-CNN-blue?style=flat-square" alt="Model">
  <img src="https://img.shields.io/badge/dataset-CIFAR--10-orange?style=flat-square" alt="Dataset">
</p>

**[Live Demo](https://image-classification-using--cnn.streamlit.app/)** • **[Report Bug](https://github.com/yourusername/image-classification-using-cnn/issues)** • **[Request Feature](https://github.com/yourusername/image-classification-using-cnn/issues)**

</div>

---

## 🎯 Overview

A production-ready Convolutional Neural Network (CNN) application that classifies images into 10 distinct categories using the CIFAR-10 dataset. Built with TensorFlow and deployed with a sleek Streamlit interface for real-time predictions.

### 📊 Supported Classes

<table>
<tr>
<td align="center">✈️ <b>Airplane</b></td>
<td align="center">🚗 <b>Automobile</b></td>
<td align="center">🐦 <b>Bird</b></td>
<td align="center">🐱 <b>Cat</b></td>
<td align="center">🦌 <b>Deer</b></td>
</tr>
<tr>
<td align="center">🐕 <b>Dog</b></td>
<td align="center">🐸 <b>Frog</b></td>
<td align="center">🐴 <b>Horse</b></td>
<td align="center">🚢 <b>Ship</b></td>
<td align="center">🚚 <b>Truck</b></td>
</tr>
</table>

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Machine Learning
- Custom CNN architecture optimized for CIFAR-10
- Data augmentation for better generalization
- Training/validation visualization
- Model checkpointing and export

</td>
<td width="50%">

### 🎨 User Interface
- Intuitive Streamlit web application
- Drag-and-drop image upload
- Real-time predictions with confidence scores
- Responsive design for all devices

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/image-classification-using-cnn.git
   cd image-classification-using-cnn
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

The app will automatically open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
image-classification-using-cnn/
│
├── 📄 app.py                    # Streamlit web application
├── 📓 model.ipynb               # Model training notebook
├── 🧠 cifar10_cnn.h5           # Pre-trained model weights
├── 🔧 inference.py              # Prediction logic
├── 📋 requirements.txt          # Python dependencies
├── 🙈 .gitignore               # Git ignore rules
└── 📖 README.md                # Project documentation
```

---

## 🎓 Training Your Own Model

Want to train from scratch or fine-tune the model? Follow these steps:

### 1. Open the Training Notebook
```bash
jupyter notebook model.ipynb
```

### 2. Training Pipeline

```python
# Load CIFAR-10 dataset (60,000 training images)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build CNN architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Train the model
history = model.fit(X_train, y_train, 
                   validation_split=0.2,
                   epochs=20, 
                   batch_size=64)

# Save the trained model
model.save('cifar10_cnn.h5')
```

### 3. Evaluate Performance

The notebook includes visualization tools to analyze:
- Training vs validation accuracy curves
- Loss progression over epochs
- Confusion matrix for test predictions
- Per-class performance metrics

---

## 💻 Usage Examples

### Web Interface

1. Navigate to the [live demo](https://image-classification-using--cnn.streamlit.app/)
2. Upload an image (JPG, JPEG, or PNG)
3. Click "Predict" to see classification results
4. View the predicted class and confidence percentage

### Programmatic Inference

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model
model = load_model("cifar10_cnn.h5")

# Load and preprocess image
img = image.load_img("sample.png", target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
predictions = model.predict(img_array)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 🛠️ Technical Stack

<table>
<tr>
<td align="center"><b>Framework</b></td>
<td align="center"><b>Library</b></td>
<td align="center"><b>Purpose</b></td>
</tr>
<tr>
<td>TensorFlow 2.x</td>
<td>Keras API</td>
<td>Model training & inference</td>
</tr>
<tr>
<td>Streamlit</td>
<td>Web framework</td>
<td>Interactive UI</td>
</tr>
<tr>
<td>NumPy</td>
<td>Array processing</td>
<td>Data manipulation</td>
</tr>
<tr>
<td>Pillow</td>
<td>Image processing</td>
<td>Image loading & preprocessing</td>
</tr>
<tr>
<td>Matplotlib</td>
<td>Visualization</td>
<td>Training metrics plotting</td>
</tr>
</table>

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~88% |
| Validation Accuracy | ~85% |
| Test Accuracy | ~83% |
| Inference Time | <100ms |
| Model Size | ~2.5MB |

*Results may vary based on training configuration and hardware*

---

## 🐛 Troubleshooting

### macOS Apple Silicon (M1/M2/M3)

If you encounter TensorFlow installation issues on Apple Silicon:

```bash
# Install using conda (recommended)
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

Or visit the [official Apple TensorFlow guide](https://developer.apple.com/metal/tensorflow-plugin/)

### Common Issues

**Issue**: Model file not found  
**Solution**: Ensure `cifar10_cnn.h5` is in the project root directory

**Issue**: Low accuracy predictions  
**Solution**: Model performs best on 32x32 images similar to CIFAR-10 training data

**Issue**: Streamlit won't start  
**Solution**: Check that port 8501 is not in use: `lsof -i :8501`

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Ideas for Contributions

- Add more CNN architectures (ResNet, VGG, etc.)
- Implement transfer learning
- Add model explainability (Grad-CAM)
- Support for additional image datasets
- Mobile app version
- Docker containerization

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit**: For making web app development effortless
- **Community**: All contributors and users of this project

---

## 📞 Contact & Support

<div align="center">

**Have questions or suggestions?**

[![GitHub Issues](https://img.shields.io/badge/Issues-GitHub-black?style=for-the-badge&logo=github)](https://github.com/yourusername/image-classification-using-cnn/issues)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail)](mailto:your.email@example.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)

### ⭐ If you found this project helpful, please give it a star!

</div>

---

<div align="center">
<sub>Built with ❤️ using TensorFlow and Streamlit</sub>
</div>
