
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_helper import load_model, predict


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("Vehicle Damage Detection")

st.write(
    "Upload a vehicle image to classify its condition "
    "into one of six categories."
)


# --------------------------------------------------
# Load and cache trained model
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()


# --------------------------------------------------
# Image uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Image processing and prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:
        # Read uploaded image directly into memory
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Vehicle Image",
            use_container_width=True
        )

        # Load model and make prediction
        with st.spinner("Analyzing vehicle image..."):

            model = get_model()

            prediction = predict(image, model)

        # Display prediction
        st.success(f"Predicted Class: {prediction}")

    except (UnidentifiedImageError, OSError):
        st.error(
            "Invalid image file. Please upload a valid JPG, JPEG or PNG image."
        )

    except Exception as e:
        st.error("Unable to process the image.")
        st.exception(e)
