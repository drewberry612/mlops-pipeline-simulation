import streamlit as st
import requests
import os
from streamlit_option_menu import option_menu
import zipfile
import io

selected = option_menu(
    menu_title=None,
    options=["Upload Images", "View Database", "Monitor Logs"],
    icons=["cloud-upload", "database", "bar-chart"],
    orientation="horizontal"
)

if selected == "Upload Images":
    # Let user upload a zip file containing images
    uploaded_zip = st.file_uploader("Upload a folder of images (as .zip)", type=["zip"])

    if uploaded_zip:
        api_url = "http://localhost:8000/upload-image/"
        # Read zip file in memory
        zip_bytes = uploaded_zip.read()
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
        image_names = [name for name in zip_file.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_names:
            st.warning("No images found in the zip file.")
        for name in image_names:
            img_data = zip_file.read(name)
            files = {"file": (os.path.basename(name), io.BytesIO(img_data), "application/octet-stream")}
            response = requests.post(api_url, files=files)
            if response.status_code == 200:
                st.success(f"Uploaded {name}")
            else:
                st.error(f"Failed to upload {name}: {response.text}")