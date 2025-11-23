import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="COVID-19 Detection from Chest X-rays",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    # Make sure best_model.h5 is in the same folder as this app.py
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

# Class names (adjust if needed)
class_names = ["Covid", "Normal", "Viral Pneumonia"]

# -------------------------------
# Header section
# -------------------------------
st.title("COVID-19 Detection from Chest X-rays using CNN")

# Show a header image (optional: put a file named 'covid_header.jpg' in the app folder)
try:
    header_img = Image.open("covid_header.jpg")
    st.image(
        header_img,
        use_column_width=True,
        caption="Deep Learning–based screening from chest X-ray images"
    )
except Exception:
    st.info("Tip: Add a file named `covid_header.jpg` in the app folder to show a header image.")

st.markdown(
    """
This app uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images as:

- 🦠 **Covid**
- ✅ **Normal**
- 🫁 **Viral Pneumonia**
    """
)

# -------------------------------
# Sidebar controls (sliders)
# -------------------------------
st.sidebar.header("Prediction Settings")

img_size = st.sidebar.slider(
    "Image size (pixels)",
    min_value=128,
    max_value=384,
    value=224,
    step=32,
    help="Input image will be resized to (size × size) before prediction."
)

threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.30,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="If maximum probability is below this, prediction will be marked as 'Uncertain'."
)

st.sidebar.markdown("---")
st.sidebar.markdown("Upload a chest X-ray image in the main panel to see predictions.")

# -------------------------------
# Helper: preprocess and predict
# -------------------------------
def preprocess_image(image: Image.Image, size: int = 224) -> np.ndarray:
    img_resized = image.resize((size, size))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(image: Image.Image, size: int):
    x = preprocess_image(image, size)
    preds = model.predict(x)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return pred_idx, confidence, preds[0]

# -------------------------------
# File uploader + main logic
# -------------------------------
st.subheader("Upload Chest X-ray Image")
uploaded_file = st.file_uploader(
    "Choose an X-ray image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    st.write("")

    if st.button("🔍 Run Prediction"):
        with st.spinner("Analyzing X-ray..."):
            pred_idx, confidence, probs = make_prediction(image, img_size)

        st.write("### Results")

        if confidence < threshold:
            st.warning(
                f"Model is **uncertain** (max confidence {confidence:.2f} < threshold {threshold:.2f}). "
                "Please interpret with caution and consult a medical professional."
            )
        else:
            st.success(
                f"**Predicted Class:** {class_names[pred_idx]}  \n"
                f"**Confidence:** {confidence:.2f}"
            )

        # Show detailed class probabilities
        st.write("#### Class probabilities")
        for cls, p in zip(class_names, probs):
            st.write(f"- **{cls}**: `{p:.3f}`")
else:
    st.info("Please upload a chest X-ray image to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool is for educational and research purposes only and **not** a substitute "
    "for professional medical diagnosis."
)
