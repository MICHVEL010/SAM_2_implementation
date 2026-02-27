import os
import cv2
import numpy as np
import torch
from typing import Tuple, List

# ─────────────────────────────────────────────
#  SAM2 model — loaded once at startup
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CHECKPOINT = os.path.join(BASE_DIR, "sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[SAM2] Loading model on {DEVICE}...")

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    _predictor = SAM2ImagePredictor(_sam2_model)
    SAM2_AVAILABLE = True
    print("[SAM2] Model loaded successfully!")
except Exception as e:
    SAM2_AVAILABLE = False
    _predictor = None
    print(f"[SAM2] Failed to load: {e}")
    print("[SAM2] Falling back to GrabCut")


# ─────────────────────────────────────────────
#  SINGLE OBJECT SEGMENTATION
# ─────────────────────────────────────────────

def segment_single_object(
    image: np.ndarray,
    point: Tuple[int, int],
    **kwargs
) -> np.ndarray:
    """
    Segment a single object using SAM2 (falls back to GrabCut if unavailable).

    Args:
        image : RGB numpy array (H, W, 3)
        point : (x, y) clicked pixel

    Returns:
        mask  : float32 (H, W), values 0.0-1.0
    """
    if SAM2_AVAILABLE:
        return _sam2_segment(image, [point])
    else:
        return _grabcut_segment(image, point)


# ─────────────────────────────────────────────
#  MULTI OBJECT SEGMENTATION
# ─────────────────────────────────────────────

def segment_multiple_objects(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    **kwargs
) -> List[np.ndarray]:
    """
    Segment multiple objects — one mask per point.

    Args:
        image  : RGB numpy array (H, W, 3)
        points : list of (x, y) clicked pixels

    Returns:
        masks  : list of float32 (H, W) masks
    """
    masks = []
    for point in points:
        try:
            mask = segment_single_object(image, point)
        except Exception:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        masks.append(mask)
    return masks


# ─────────────────────────────────────────────
#  SAM2 CORE
# ─────────────────────────────────────────────

def _sam2_segment(
    image: np.ndarray,
    points: List[Tuple[int, int]],
) -> np.ndarray:
    """Run SAM2 prediction for a single point and return best mask."""
    global _predictor

    input_point = np.array(points)
    input_label = np.ones(len(points), dtype=np.int32)

    _predictor.set_image(image)
    masks, scores, _ = _predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Pick the mask with the highest confidence score
    best = np.argmax(scores)
    return masks[best].astype(np.float32)


# ─────────────────────────────────────────────
#  GRABCUT FALLBACK
# ─────────────────────────────────────────────

def _grabcut_segment(
    image: np.ndarray,
    point: Tuple[int, int],
    iterations: int = 7,
    refine_kernel_size: int = 3,
) -> np.ndarray:
    """GrabCut fallback if SAM2 is not available."""
    h, w = image.shape[:2]
    px, py = int(point[0]), int(point[1])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = max(2, int(min(w, h) * 0.02))
    cv2.grabCut(bgr, mask, (m, m, w-2*m, h-2*m), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    cv2.circle(mask, (px, py), max(15, min(w, h) // 6), cv2.GC_FGD, -1)
    for cx, cy in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
        cv2.circle(mask, (cx, cy), max(3, min(w, h) // 15), cv2.GC_BGD, -1)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size*10, refine_kernel_size*10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 1, thickness=cv2.FILLED)
    return cv2.GaussianBlur(binary.astype(np.float32), (7, 7), 0)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB + float mask → RGBA (transparent background)."""
    alpha = (mask * 255).astype(np.uint8)
    return np.dstack([image, alpha])


def overlay_masks_on_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Draws semi-transparent coloured overlays for each mask."""
    COLORS = [(255,60,60),(60,180,60),(60,100,255),(255,200,0),(200,0,255),(0,220,220)]
    result = image.copy().astype(np.float32)
    for i, mask in enumerate(masks):
        c = COLORS[i % len(COLORS)]
        for j in range(3):
            result[:,:,j] = np.where(
                mask > 0.5,
                result[:,:,j] * (1-alpha) + c[j] * alpha,
                result[:,:,j]
            )
    return np.clip(result, 0, 255).astype(np.uint8)