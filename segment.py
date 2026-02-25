import cv2
import numpy as np
from typing import Tuple, List

MAX_DIM_SINGLE = 600
MAX_DIM_MULTI = 400


def segment_single_object(image, point, margin_percent=0.02, iterations=7, refine_kernel_size=3):
    orig_h, orig_w = image.shape[:2]
    px, py = int(point[0]), int(point[1])
    scale = min(MAX_DIM_SINGLE / orig_w, MAX_DIM_SINGLE / orig_h, 1.0)
    if scale < 1.0:
        small = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        spx, spy = int(px * scale), int(py * scale)
    else:
        small, spx, spy = image.copy(), px, py
    h, w = small.shape[:2]
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = max(2, int(min(w, h) * margin_percent))
    cv2.grabCut(bgr, mask, (m, m, w - 2*m, h - 2*m), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    cv2.circle(mask, (spx, spy), max(15, min(w, h) // 6), cv2.GC_FGD, -1)
    for cx, cy in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        cv2.circle(mask, (cx, cy), max(3, min(w, h) // 15), cv2.GC_BGD, -1)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")
    binary = _colour_filter(small, binary, (spx, spy))
    binary = _keep_component(binary, (spx, spy))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size * 10, refine_kernel_size * 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    # Fill all internal holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 1, thickness=cv2.FILLED)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size, refine_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2)
    if scale < 1.0:
        binary = cv2.resize(binary.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (binary > 0.5).astype(np.uint8)
    return cv2.GaussianBlur(binary.astype(np.float32), (7, 7), 0)


def segment_multiple_objects(image, points, margin_percent=0.02, iterations=3, refine_kernel_size=3):
    orig_h, orig_w = image.shape[:2]
    scale = min(MAX_DIM_MULTI / orig_w, MAX_DIM_MULTI / orig_h, 1.0)
    if scale < 1.0:
        small = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        scaled_points = [(int(px * scale), int(py * scale)) for px, py in points]
    else:
        small, scaled_points = image, list(points)
    masks = []
    for point in scaled_points:
        try:
            mask_small = _grabcut_single(small, point, margin_percent, iterations, refine_kernel_size)
            if scale < 1.0:
                mask_up = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                mask_up = (mask_up > 0.5).astype(np.float32)
            else:
                mask_up = mask_small
        except Exception:
            mask_up = np.zeros((orig_h, orig_w), dtype=np.float32)
        masks.append(mask_up)
    return masks


def _grabcut_single(image, point, margin_percent=0.02, iterations=3, refine_kernel_size=3):
    h, w = image.shape[:2]
    px, py = int(point[0]), int(point[1])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    m = max(2, int(min(w, h) * margin_percent))
    cv2.grabCut(bgr, mask, (m, m, w - 2*m, h - 2*m), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    cv2.circle(mask, (px, py), max(15, min(w, h) // 6), cv2.GC_FGD, -1)
    for cx, cy in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        cv2.circle(mask, (cx, cy), max(3, min(w, h) // 15), cv2.GC_BGD, -1)
    cv2.grabCut(bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")
    binary = _colour_filter(image, binary, (px, py))
    binary = _keep_component(binary, (px, py))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size * 10, refine_kernel_size * 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    # Fill all internal holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 1, thickness=cv2.FILLED)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (refine_kernel_size, refine_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k2)
    return cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)


def _colour_filter(image, binary_mask, point, tolerance=80):
    px, py = point
    click_colour = image[py, px].astype(np.float32)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    result = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        pixels = image[labels == label].astype(np.float32)
        if len(pixels) == 0:
            continue
        if float(np.linalg.norm(pixels.mean(axis=0) - click_colour)) <= tolerance:
            result[labels == label] = 1
    clicked_label = labels[py, px]
    if clicked_label > 0:
        result[labels == clicked_label] = 1
    return result


def _keep_component(binary_mask, point):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    px, py = point
    clicked_label = labels[py, px]
    if clicked_label > 0:
        return (labels == clicked_label).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return (labels == int(np.argmax(areas)) + 1).astype(np.uint8)


def apply_mask_to_image(image, mask):
    alpha = (mask * 255).astype(np.uint8)
    return np.dstack([image, alpha])


def overlay_masks_on_image(image, masks, alpha=0.45):
    COLORS = [(255, 60, 60), (60, 180, 60), (60, 100, 255), (255, 200, 0), (200, 0, 255), (0, 220, 220)]
    result = image.copy().astype(np.float32)
    for i, mask in enumerate(masks):
        c = COLORS[i % len(COLORS)]
        for j in range(3):
            result[:, :, j] = np.where(mask > 0.5, result[:, :, j] * (1 - alpha) + c[j] * alpha, result[:, :, j])
    return np.clip(result, 0, 255).astype(np.uint8)