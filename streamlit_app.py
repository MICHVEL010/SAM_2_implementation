"""
streamlit_app.py  –  Streamlit Frontend
----------------------------------------
Provides a full UI for:
  • Single-object extraction  (click once → get transparent PNG)
  • Multi-object extraction   (click multiple points → get ZIP)
  • Coloured overlay preview  (see all masks at once)

Talks to the FastAPI backend running on http://localhost:8000
"""

import io
import json
import zipfile

import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

API_BASE   = "http://localhost:8000"
PAGE_TITLE = "🖼️ Object Extraction App"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# ─────────────────────────────────────────────
#  Sidebar – mode selector + instructions
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Extraction mode",
        ["Single Object", "Multi Object"],
        help="Single: click once and extract.\n"
             "Multi: click several points then press Extract.",
    )
    st.markdown("---")
    st.markdown(
        "**How to use**\n"
        "1. Upload an image\n"
        "2. Click the object(s) you want\n"
        "3. Press the Extract button\n"
        "4. Download the result(s)"
    )
    st.markdown("---")
    if st.button("🔄 Check API health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.status_code == 200:
                st.success("API is online ✅")
            else:
                st.error(f"API returned {r.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Is it running?\n`uvicorn app:app --reload`")


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────

def make_checkerboard(width: int, height: int, square: int = 20) -> Image.Image:
    """
    Creates a grey/white checkerboard background (like Photoshop).
    Makes transparent areas visible without hiding dark pixels.
    """
    light = (200, 200, 200)
    dark  = (150, 150, 150)
    board = Image.new("RGB", (width, height))
    for y in range(0, height, square):
        for x in range(0, width, square):
            color = light if (x // square + y // square) % 2 == 0 else dark
            for py in range(y, min(y + square, height)):
                for px in range(x, min(x + square, width)):
                    board.putpixel((px, py), color)
    return board


def place_on_checkerboard(rgba_img: Image.Image) -> Image.Image:
    """Paste an RGBA image onto a checkerboard background."""
    board = make_checkerboard(rgba_img.width, rgba_img.height)
    board = board.convert("RGBA")
    board.paste(rgba_img, mask=rgba_img.split()[3])
    return board

def draw_points_on_image(
    image: Image.Image,
    points: list,
    radius: int = 10,
    line_len: int = 20,
) -> Image.Image:
    """Draw red crosshair + circle for every clicked point."""
    marked = image.copy()
    draw   = ImageDraw.Draw(marked)
    COLORS = ["red", "lime", "blue", "yellow", "magenta", "cyan"]

    for i, (px, py) in enumerate(points):
        color = COLORS[i % len(COLORS)]
        draw.ellipse(
            [(px - radius, py - radius), (px + radius, py + radius)],
            outline=color, width=2,
        )
        draw.line([(px - line_len, py), (px + line_len, py)], fill=color, width=2)
        draw.line([(px, py - line_len), (px, py + line_len)], fill=color, width=2)
        draw.text((px + radius + 2, py - radius), str(i + 1), fill=color)

    return marked


def call_single_extract(image_bytes: bytes, filename: str, x: int, y: int):
    """POST /extract/single → bytes or None."""
    resp = requests.post(
        f"{API_BASE}/extract/single",
        files={"file": (filename, image_bytes, "image/jpeg")},
        data={"x": x, "y": y},
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.content
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"API error {resp.status_code}: {detail}")
    return None


def call_multi_extract(image_bytes: bytes, filename: str, points: list):
    """POST /extract/multi → ZIP bytes or None."""
    resp = requests.post(
        f"{API_BASE}/extract/multi",
        files={"file": (filename, image_bytes, "image/jpeg")},
        data={"points": json.dumps(points)},
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.content
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"API error {resp.status_code}: {detail}")
    return None


def call_preview(image_bytes: bytes, filename: str, points: list):
    """POST /extract/preview → PNG bytes or None."""
    resp = requests.post(
        f"{API_BASE}/extract/preview",
        files={"file": (filename, image_bytes, "image/jpeg")},
        data={"points": json.dumps(points)},
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.content
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"API error {resp.status_code}: {detail}")
    return None


# ─────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible"
)

if uploaded_file is None:
    st.info("👆 Please upload an image to get started.")
    st.stop()

# ── Load image ───────────────────────────────
image_bytes = uploaded_file.read()
image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
orig_w, orig_h = image.size

# ── Display width & scale factor ─────────────
DISPLAY_W   = 800
scale       = orig_w / DISPLAY_W   # maps display-px → actual-px

# ═════════════════════════════════════════════
#  MODE: SINGLE OBJECT
# ═════════════════════════════════════════════

if mode == "Single Object":
    st.subheader("🎯 Single Object Extraction")
    st.write("**Click on the object you want to extract:**")

    value = streamlit_image_coordinates(image, width=DISPLAY_W, key="single_pick")

    if value is not None:
        actual_x = int(value["x"] * scale)
        actual_y = int(value["y"] * scale)

        # Show marked preview
        marked = draw_points_on_image(image, [(actual_x, actual_y)])
        st.image(marked, caption=f"Clicked: ({actual_x}, {actual_y})", use_column_width=True)

        if st.button("✂️ Extract Object", type="primary"):
            with st.spinner("Extracting... please wait"):
                result_bytes = call_single_extract(
                    image_bytes, uploaded_file.name, actual_x, actual_y
                )

            if result_bytes:
                result_img = Image.open(io.BytesIO(result_bytes))

                st.markdown("---")
                st.subheader("✅ Extracted Object")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    checker = place_on_checkerboard(result_img)
                    st.image(checker, caption="Extracted (checkerboard = transparent)", use_column_width=True)

                st.download_button(
                    label="📥 Download PNG (transparent)",
                    data=result_bytes,
                    file_name="extracted_object.png",
                    mime="image/png",
                )
    else:
        st.info("👆 Click on the image above to select a point, then press Extract.")


# ═════════════════════════════════════════════
#  MODE: MULTI OBJECT
# ═════════════════════════════════════════════

else:
    st.subheader("🎯 Multi-Object Extraction")
    st.write("**Click on each object you want to extract. Then press Extract.**")

    # Session-state list of clicked points
    if "multi_points" not in st.session_state:
        st.session_state.multi_points = []

    col_img, col_info = st.columns([3, 1])

    with col_img:
        # Draw existing points on the image before showing picker
        preview_img = draw_points_on_image(image, st.session_state.multi_points)
        value = streamlit_image_coordinates(preview_img, width=DISPLAY_W, key="multi_pick")

    with col_info:
        st.markdown("**Clicked points:**")
        if st.session_state.multi_points:
            for i, (px, py) in enumerate(st.session_state.multi_points):
                st.write(f"  {i+1}. ({px}, {py})")
        else:
            st.write("_None yet_")

        if st.button("🗑️ Clear all points"):
            st.session_state.multi_points = []
            st.rerun()

    # Register new click (only add if it's a new coordinate)
    if value is not None:
        new_x = int(value["x"] * scale)
        new_y = int(value["y"] * scale)
        coord = (new_x, new_y)
        if coord not in st.session_state.multi_points:
            st.session_state.multi_points.append(coord)
            st.rerun()

    # ── Action buttons ────────────────────────
    st.markdown("---")
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        preview_btn = st.button(
            "👁️ Preview Masks",
            disabled=len(st.session_state.multi_points) == 0,
        )

    with btn_col2:
        extract_btn = st.button(
            "✂️ Extract All Objects",
            type="primary",
            disabled=len(st.session_state.multi_points) == 0,
        )

    # ── Preview ───────────────────────────────
    if preview_btn and st.session_state.multi_points:
        with st.spinner("Generating preview…"):
            prev_bytes = call_preview(
                image_bytes,
                uploaded_file.name,
                st.session_state.multi_points,
            )
        if prev_bytes:
            st.subheader("🎨 Mask Overlay Preview")
            st.image(prev_bytes, caption="Colour-coded masks", use_column_width=True)

    # ── Extract ───────────────────────────────
    if extract_btn and st.session_state.multi_points:
        with st.spinner("Extracting all objects… this may take a moment"):
            png_bytes = call_multi_extract(
                image_bytes,
                uploaded_file.name,
                st.session_state.multi_points,
            )

        if png_bytes:
            st.markdown("---")
            st.subheader("✅ Extracted Objects")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                result_img = Image.open(io.BytesIO(png_bytes))
                display_img = place_on_checkerboard(result_img) if result_img.mode == "RGBA" else result_img
                st.image(display_img, caption="All objects extracted", use_column_width=True)

            st.download_button(
                label="📥 Download PNG (transparent)",
                data=png_bytes,
                file_name="extracted_objects.png",
                mime="image/png",
            )

    if not st.session_state.multi_points:
        st.info("👆 Click on objects in the image above, then press Preview or Extract.")