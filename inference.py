import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load trained model
model = load_model("cifar10_cnn.keras")

def preprocess_image(image: Image.Image):
    image = image.resize((32, 32))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image: Image.Image):
    img = preprocess_image(image)
    logits = model.predict(img)
    probs = tf.nn.softmax(logits).numpy()[0]
    pred_class = CLASS_NAMES[np.argmax(probs)]
    confidence = np.max(probs)
    return pred_class, confidence
