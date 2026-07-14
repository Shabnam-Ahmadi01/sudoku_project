"""
Phase 4b: Draw the solved digits back onto the ORIGINAL photo (not the
warped square), by inverse-warping an overlay image through the same
perspective transform Phase 1 used, then alpha-compositing it onto the
original.

Given digits (already visible in the photo) are left alone; only the
solver-filled cells get drawn, in a different color, so the result reads
as "here's what goes in the blanks" rather than redrawing the whole grid.
"""

import cv2
import numpy as np
from config import WARP_SIZE


def _solved_cell_positions(given_matrix, solved_matrix):
    """Returns list of (row, col, digit) for cells that were empty in the
    recognized input but filled in by the solver."""
    filled = []
    for r in range(9):
        for c in range(9):
            if given_matrix[r][c] == 0:
                filled.append((r, c, solved_matrix[r][c]))
    return filled


def render_overlay_square(given_matrix, solved_matrix, size=WARP_SIZE,
                           color=(40, 200, 40)):
    """Renders a size x size BGRA image (transparent background) with the
    solver-filled digits drawn in the correct cell positions, in the
    given color (default green)."""
    overlay = np.zeros((size, size, 4), dtype=np.uint8)
    step = size / 9.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for r, c, digit in _solved_cell_positions(given_matrix, solved_matrix):
        cx = int(c * step + step / 2)
        cy = int(r * step + step / 2)

        text = str(digit)
        font_scale = step / 45.0
        thickness = max(2, int(step / 25))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        origin = (cx - tw // 2, cy + th // 2)

        cv2.putText(overlay, text, origin, font, font_scale,
                    (*color, 255), thickness, cv2.LINE_AA)

    return overlay


def warp_overlay_to_original(overlay_bgra, original_shape, corners,
                              size=WARP_SIZE):
    """Inverse-warps the square overlay back into the original image's
    geometry using the same 4 corners Phase 1 detected."""
    dst = np.array([
        [0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]
    ], dtype=np.float32)
    # inverse of corners -> square is square -> corners
    M_inv = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    h, w = original_shape[:2]
    warped_overlay = cv2.warpPerspective(overlay_bgra, M_inv, (w, h))
    return warped_overlay


def composite_overlay(original_bgr, warped_overlay_bgra):
    """Alpha-composites the warped overlay onto the original image."""
    alpha = warped_overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = warped_overlay_bgra[:, :, :3].astype(np.float32)
    bg = original_bgr.astype(np.float32)
    out = fg * alpha + bg * (1 - alpha)
    return out.astype(np.uint8)


def draw_solution_on_original(original_bgr, corners, given_matrix,
                               solved_matrix, color=(40, 200, 40)):
    """Full pipeline: given the original photo, the detected grid corners
    (from Phase 1), the recognized matrix (0=empty), and the solved
    matrix, returns a copy of the original photo with the solved digits
    drawn in on top."""
    overlay_sq = render_overlay_square(given_matrix, solved_matrix, color=color)
    warped_overlay = warp_overlay_to_original(overlay_sq, original_bgr.shape, corners)
    result = composite_overlay(original_bgr, warped_overlay)
    return result