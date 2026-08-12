
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os
import yaml

# --- Configuration ---
# Path to the trained YOLOv8 model weights
MODEL_PATH = "best.pt"
# Path to the data.yaml file, which contains class names
DATA_YAML_PATH = "data.yaml"

# --- Load Model and Classes ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model and caches it."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        st.stop()

@st.cache_data
def load_classes(yaml_path):
    """Loads class names from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', ['unknown'])
    except Exception as e:
        st.error(f"Error loading classes from {yaml_path}: {e}")
        st.stop()

model = load_model(MODEL_PATH)
class_names = load_classes(DATA_YAML_PATH)

# --- Streamlit App ---
st.title("YOLOv8 Traffic Object Detection App")
st.write("Upload an image to detect 'car' objects using a trained YOLOv8n model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # Load image for inference
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    # Perform inference
    results = model(image)

    # Process and display results
    for r in results:
        # Plot results on the image (returns a BGR numpy array)
        im_array = r.plot()
        # Convert BGR to RGB for PIL and Streamlit
        processed_image = Image.fromarray(im_array[..., ::-1])
        st.image(processed_image, caption="Detected Objects", use_column_width=True)

        # Optional: display detected objects and their confidence
        st.subheader("Detected Objects Summary")
        if len(r.boxes) == 0:
            st.write("No objects detected.")
        else:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # Ensure class_id is within bounds of class_names
                label = class_names[class_id] if class_id < len(class_names) else f"Unknown Class {class_id}"
                st.write(f"- **{label}**: Confidence {confidence:.2f}")

st.markdown("""
---
### Deployment Instructions:
1.  **Save these files:** Ensure `app.py`, `best.pt`, `data.yaml`, and `requirements.txt` are in the same folder (`Deployment/`).
2.  **Create a GitHub Repository:** Push the `Deployment/` folder and its contents to a new public GitHub repository.
3.  **Deploy on Streamlit Cloud:** Go to [Streamlit Cloud](https://share.streamlit.io/) and select \"New app\". Connect your GitHub repository and choose the `app.py` file within your `Deployment` folder as the main file.
""")