import streamlit as st
import requests
import os
from PIL import Image
import numpy as np

# Configuration
API_URL = "http://localhost:8000/predict"
IMAGE_FOLDER = "webapp/data/images"  # Replace with your actual image folder path
MASK_FOLDER = "webapp/data/masks"  # Replace with your actual mask folder path

st.title("Image Segmentation App")


# Function to display images
def display_image(image_path, caption):
    try:
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=True)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {e}")


# Get image filenames from folder
image_filenames = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]

# Dropdown to select an image
selected_image = st.selectbox("Select an Image:", image_filenames)

if selected_image:
    # Construct full image paths
    image_path = os.path.join(IMAGE_FOLDER, selected_image)
    mask_path = os.path.join(MASK_FOLDER, selected_image[:-len("_leftImg8bit.png")] + "_gtFine_labelIds.png")

    # Display original image
    display_image(image_path, "Original Image")

    # Display ground truth mask
    display_image(mask_path, "Ground Truth Mask")

    # Make API call
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            # predicted_mask = np.array(response.json()['mask'])  # Assuming API returns a JSON with a 'mask' key
            st.image(response.content, caption="Predicted Mask", use_container_width=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Error during API request: {e}")
    except KeyError:
        st.error(f"API response does not contain the 'mask' key")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.warning("Please select an image from the dropdown.")