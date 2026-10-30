import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Medicinal Plant AI",
    page_icon="🌿"
)

st.title("🌿 Medicinal Plant Identification")
st.write("Upload a leaf photo to identify Neem, Tulsi or Aloe Vera.")

@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model(
        "medicinal_plant_model.keras"
    )

model = load_plant_model()

class_names = ["Aloe Vera", "Neem", "Tulsi"]

plant_uses = {
    "Aloe Vera": "Commonly used in skin-care products.",
    "Neem": "Traditionally used in skin-care preparations.",
    "Tulsi": "Traditionally used in herbal preparations."
}

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Plant Image")

    resized = image.resize((160, 160))
    image_array = np.array(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    index = int(np.argmax(prediction[0]))
    plant_name = class_names[index]
    confidence = float(np.max(prediction[0]) * 100)

    st.success(f"Predicted Plant: {plant_name}")
    st.metric("Model Confidence", f"{confidence:.2f}%")
    st.write("Traditional Uses:", plant_uses[plant_name])
    st.caption("For educational purposes only.")
