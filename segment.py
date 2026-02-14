# segment.py
import cv2
import numpy as np
from typing import Tuple
import sys

def extract_object(
    image: np.ndarray,
    point: Tuple[int, int],
    margin_percent: float = 0.1,
    iterations: int = 5,
    refine_kernel_size: int = 5
) -> np.ndarray:
    """
    Extracts an object from an image using GrabCut algorithm.
    
    Args:
        image: Input image in RGB format (numpy array)
        point: Center point (x, y) of the object to extract
        margin_percent: Margin percentage for initial rectangle (default: 0.1)
        iterations: Number of GrabCut iterations (default: 5)
        refine_kernel_size: Kernel size for morphological operations (default: 5)
    
    Returns:
        Extracted object as RGBA image (numpy array with alpha channel)
    """
    # Initialize GrabCut models
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Calculate rectangle around the point
    h, w = image.shape[:2]
    margin_w = int(w * margin_percent)
    margin_h = int(h * margin_percent)
    rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
    
    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Refine edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size, refine_kernel_size))
    mask_refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask_refined = cv2.GaussianBlur(mask_refined.astype(np.float32), (5, 5), 0)
    
    # Create alpha channel from refined mask
    alpha = (mask_refined * 255).astype(np.uint8)
    
    # Create RGBA image (RGB + Alpha channel)
    extracted_object = np.dstack([image, alpha])
    
    return extracted_object


if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("Object Extraction Tool")
    print("=" * 60)
    
    # Get image path from user
    if len(sys.argv) > 1:
        # Command line argument
        image_path = sys.argv[1]
    else:
        # Interactive input
        image_path = input("Enter image path (e.g., source.jpg): ").strip()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        sys.exit(1)
    
    # Load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Cannot read image '{image_path}'")
            sys.exit(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # Get coordinates from user
    if len(sys.argv) > 3:
        # Command line arguments
        x = int(sys.argv[2])
        y = int(sys.argv[3])
    else:
        # Interactive input
        print(f"\nImage size: {img.shape[1]} x {img.shape[0]} (width x height)")
        print("Enter coordinates of the object:")
        
        try:
            x = int(input("  X coordinate: ").strip())
            y = int(input("  Y coordinate: ").strip())
        except ValueError:
            print("❌ Error: Coordinates must be numbers!")
            sys.exit(1)
    
    # Validate coordinates
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        print(f"❌ Error: Coordinates ({x}, {y}) are out of bounds!")
        print(f"   Valid range: X: 0-{img.shape[1]-1}, Y: 0-{img.shape[0]-1}")
        sys.exit(1)
    
    point = (x, y)
    print(f"✓ Using point: ({x}, {y})")
    
    # Extract object
    print("\nExtracting object... (this may take a few seconds)")
    extracted_object = extract_object(img, point)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_extracted.png"

    # Save extracted object
    cv2.imwrite(output_path, cv2.cvtColor(extracted_object, cv2.COLOR_RGBA2BGRA))
    
    print(f"\n✓ SUCCESS!")
    print(f"✓ Extracted object saved to: {output_path}")
    print(f"✓ Object shape: {extracted_object.shape} (height, width, RGBA)")
    print("=" * 60)