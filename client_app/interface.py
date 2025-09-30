import streamlit as st
import requests

def main():
    st.title("Image Classification Client")
    st.write("Upload an image to get top 3 predicted classes.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify Image"):
            files = {"file": uploaded_file.getvalue()}
            # Replace with API endpoint
            api_url = "http://localhost:8000/predict"
            try:
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                result = response.json()
                st.write("Top 3 Predictions:")
                for pred in result.get("predictions", []):
                    st.write(f"Label: {pred['label']}, Probability: {pred['probability']:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()