# First, let's create the function file
import cv2
import numpy as np
from typing import Tuple

def segment_object_from_point(
    image: np.ndarray,
    point: Tuple[int, int],
    margin_percent: float = 0.1,
    iterations: int = 5,
    refine_kernel_size: int = 5
) -> np.ndarray:
    """
    Segments an object from an image using GrabCut algorithm.
    
    Args:
        image: Input image in RGB format (numpy array)
        point: Center point (x, y) of the object to segment
        margin_percent: Margin percentage for initial rectangle (default: 0.1)
        iterations: Number of GrabCut iterations (default: 5)
        refine_kernel_size: Kernel size for morphological operations (default: 5)
    
    Returns:
        Refined binary mask (numpy array, values 0-1 as float32)
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
    
    return mask_refined

# Test it
source_image = cv2.imread('source.jpg')
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

# Get center point
h, w = source_image.shape[:2]
center_point = (w // 2, h // 2)

# Segment
mask = segment_object_from_point(source_image, center_point)

# Save mask
cv2.imwrite('mask_output.png', (mask * 255).astype(np.uint8))

print(f"✓ Mask created! Shape: {mask.shape}")
print(f"✓ Saved to: mask_output.png")