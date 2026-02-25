import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Image Coordinate Picker", layout="wide")
st.title("🖼️ Image Coordinate Picker")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and prepare the image
    image = Image.open(uploaded_file)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    orig_width, orig_height = image.size
    
    st.write("**Click on the image to select coordinates:**")
    
    # Display the image with coordinate picking
    display_width = 800
    value = streamlit_image_coordinates(
        image,
        width=display_width,
        key="image"
    )
    
    # Show results after clicking
    if value is not None:
        # Calculate actual coordinates
        scale_factor = orig_width / display_width
        actual_x = int(value["x"] * scale_factor)
        actual_y = int(value["y"] * scale_factor)
        
        st.markdown("---")
        st.success(f"✅ **Selected Coordinates:** X = {actual_x}, Y = {actual_y}")
        
        # Get pixel color
        try:
            pixel = image.getpixel((actual_x, actual_y))
            if isinstance(pixel, tuple):
                st.write(f"**Pixel Color (RGB):** R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")
        except:
            pass
        
        st.markdown("---")
        st.subheader("📍 Image with Marked Point")
        
        # Create a copy and mark the point
        marked_image = image.copy()
        draw = ImageDraw.Draw(marked_image)
        
        # Draw red circle at the clicked point
        radius = 10
        draw.ellipse(
            [(actual_x - radius, actual_y - radius),
             (actual_x + radius, actual_y + radius)],
            fill='red',
            outline='red'
        )
        
        # Draw crosshair
        line_length = 20
        draw.line(
            [(actual_x - line_length, actual_y), (actual_x + line_length, actual_y)],
            fill='red',
            width=2
        )
        draw.line(
            [(actual_x, actual_y - line_length), (actual_x, actual_y + line_length)],
            fill='red',
            width=2
        )
        
        # Show the marked image
        st.image(
            marked_image,
            caption=f"Point marked at coordinates ({actual_x}, {actual_y})",
            use_container_width=True
        )
        
        # Download button
        from io import BytesIO
        buf = BytesIO()
        marked_image.save(buf, format="PNG")
        
        st.download_button(
            label="📥 Download Marked Image",
            data=buf.getvalue(),
            file_name="image_with_point.png",
            mime="image/png"
        )
    
    else:
        st.info("👆 Click on the image above to select a point")

else:
    st.info("👆 Please upload an image to get started!")