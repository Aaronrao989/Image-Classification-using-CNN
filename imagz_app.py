
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🧠 Image Classification using CNN")
st.write("Upload an image and get predicted class from CIFAR-10 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model("model.h5")
    image_array = np.array(image) / 255.0
    prediction = model.predict(np.expand_dims(image_array, axis=0))

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
