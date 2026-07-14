"""
Phase 1a: Grid detection + perspective transform.

Handles input images of ANY aspect ratio / resolution. Detection is done on a
resized-for-speed copy of the image; contour coordinates are scaled back up to
the original resolution before warping, so we never lose precision from the
downscale. The output of warp_to_square() is always a fixed WARP_SIZE x
WARP_SIZE image, so every stage after this one is ratio-agnostic.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from config import WARP_SIZE

DETECT_MAX_DIM = 1200     # max dimension used for contour search (precision
                           # vs speed tradeoff -- too low causes corner
                           # error that gets amplified after warping)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def debug_imshow(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    
def _resize_for_detection(img):
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    if scale < 1.0:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
    else:
        resized = img.copy()
        scale = 1.0
    return resized, scale


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        15, 5
    )
    # morphological CLOSE (dilate then erode) closes small gaps in grid
    # lines WITHOUT growing the outer boundary outward the way a plain
    # dilate would -- a plain dilate here was shifting detected corners by
    # tens of pixels after upscaling, badly misaligning cell boundaries.
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, thresh


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_grid_contour(thresh, resized_shape, scale):
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = resized_shape[0] * resized_shape[1]
    best = None
    best_area = 0

    # Area threshold is intentionally low (5%) -- in real photos the grid
    # may occupy anywhere from a small portion to nearly the whole frame,
    # and we don't want to reject valid detections just because the grid
    # is small in the shot. The 4-point + aspect-ratio checks below filter
    # out non-grid contours instead.
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue
        if area > best_area:
            best = c
            best_area = area

    if best is None:
        return None

    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        # fallback: use minAreaRect to get a 4-point box even if the contour
        rect = cv2.minAreaRect(best)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            return None
        aspect = rw / rh
        if not (0.7 <= aspect <= 1.3):
            return None   # not roughly square -> probably not the grid

        pts = cv2.boxPoints(rect)  

    corners = order_points(pts) / scale
    return corners


def warp_to_square(img, corners, size=WARP_SIZE):
    """Perspective-warp the detected quadrilateral to a fixed size x size
    square, regardless of the original image's aspect ratio."""
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped


def detect_and_warp(path, debug=False):
    """Full phase-1a pipeline: load -> detect grid -> warp to fixed square.
    Returns (warped_bgr, warped_gray, corners) or raises ValueError if no
    grid was found."""
    img = load_image(path)
    resized, scale = _resize_for_detection(img)
    _, thresh = preprocess(resized)
    corners = find_grid_contour(thresh, resized.shape[:2], scale)

    if corners is None:
        raise ValueError(f"No sudoku grid contour found in {path}")

    warped = warp_to_square(img, corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if debug:
        return warped, warped_gray, corners, thresh
    return warped, warped_gray, corners
