
import sys
!pip install -q streamlit
!pip install -q keras-tuner
!pip install -q kaggle

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import models, layers
import os
import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import kagglehub

# --- Data Loading and Preprocessing (copied from previous cells) ---

# Download dataset
path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")

# Load and preprocess images
image_size = (224, 224)
base_dir = os.path.join(path, "Covid19-dataset", "train")
classes = ["Covid", "Normal", "Viral Pneumonia"]
images = []
labels = []

for cls in classes:
    cls_folder = os.path.join(base_dir, cls)
    if not os.path.exists(cls_folder):
        print(f"Warning: Class folder not found: {cls_folder}. Skipping.")
        continue
    for img_path in glob.glob(os.path.join(cls_folder, "*")):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}. Skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        images.append(img)
        labels.append(cls)

X = np.array(images, dtype="float32")
y = np.array(labels)

# Normalize pixel values
X_norm = X / 255.0

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Train/Validation/Test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X_norm, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
)

# One-hot encode labels for CNNs
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- Keras Tuner Setup ---
input_shape = X_train.shape[1:]

def model_builder(hp):
    hp_filters = hp.Choice('filters', values=[32, 64])
    hp_units = hp.Choice('dense_units', values=[128, 256])
    hp_dropout = hp.Choice('dropout', values=[0.3, 0.5])

    model = models.Sequential()
    model.add(layers.Conv2D(hp_filters, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(hp_filters*2, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp_units, activation='relu'))
    model.add(layers.Dropout(hp_dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='kt_covid',
    project_name='basic_cnn_tuning'
)

tuner.search(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=5,
    verbose=1
)

# Get the best model from the Keras Tuner search
best_model = tuner.get_best_models(num_models=1)[0]
# Save the best model
best_model.save("best_model.h5")

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
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

# Class names (adjust if needed)
class_names = ["Covid", "Normal", "Viral Pneumonia"]

# -------------------------------
# Header section
# -------------------------------
st.title("COVID-19 Detection from Chest X-rays using CNN")

# Show a header image (put a file named 'covid_header.jpg' in the app folder)
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
# File uploader
# -------------------------------
st.subheader("Upload Chest X-ray Image")
uploaded_file = st.file_uploader(
    "Choose an X-ray image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

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
# Main logic
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    st.write("")

    if st.button(" 🔍 Run Prediction"):
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
