# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

st.set_page_config(page_title="Object Extractor", layout="wide")

st.title("🎯 Object Extraction Tool")
st.markdown("Upload an image and click on the object you want to extract")

# API endpoint
API_URL = "http://localhost:8000/extract"

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        # Show image dimensions
        st.info(f"Image size: {image.width} x {image.height} pixels")
    
    # Input coordinates
    st.subheader("📍 Enter Object Coordinates")
    
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        x = st.number_input(
            "X coordinate",
            min_value=0,
            max_value=image.width - 1,
            value=image.width // 2,
            help="Horizontal position (0 to image width)"
        )
    
    with coord_col2:
        y = st.number_input(
            "Y coordinate",
            min_value=0,
            max_value=image.height - 1,
            value=image.height // 2,
            help="Vertical position (0 to image height)"
        )
    
    # Show selected point on image
    if st.checkbox("Show selected point on image", value=True):
        img_with_point = img_array.copy()
        # Draw circle at selected point
        import cv2
        cv2.circle(img_with_point, (int(x), int(y)), 10, (255, 0, 0), -1)
        cv2.circle(img_with_point, (int(x), int(y)), 12, (255, 255, 255), 2)
        st.image(img_with_point, caption=f"Selected point: ({x}, {y})", use_container_width=True)
    
    # Extract button
    if st.button("🚀 Extract Object", type="primary"):
        with st.spinner("Extracting object..."):
            try:
                # Prepare request
                uploaded_file.seek(0)  # Reset file pointer
                files = {"file": uploaded_file.getvalue()}
                data = {"x": x, "y": y}
                
                # Send request to API
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    # Display extracted object
                    extracted_image = Image.open(io.BytesIO(response.content))
                    
                    with col2:
                        st.subheader("Extracted Object")
                        st.image(extracted_image, use_container_width=True)
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Extracted Object",
                            data=response.content,
                            file_name="extracted_object.png",
                            mime="image/png"
                        )
                    
                    st.success("✅ Object extracted successfully!")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload an image to get started")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Upload an image** using the file uploader
    2. **Enter coordinates** of the object you want to extract
       - Use the image dimensions as guide
       - Or use center point as default
    3. **Click "Extract Object"** to process
    4. **Download** the extracted object with transparent background
    
    ---
    
    ### 💡 Tips:
    - Coordinates start at (0, 0) in top-left corner
    - X increases going right
    - Y increases going down
    - Enable "Show selected point" to verify position
    
    ---
    
    ### 🔧 API Status
    """)
    
    # Check API health
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API is running")
        else:
            st.error("❌ API not responding")
    except:
        st.error("❌ API not running")
        st.markdown("Start API with: `uvicorn app:app --reload`")