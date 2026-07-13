"""
Phase 1b: Split the warped 900x900 grid into 81 cells and flag which are
empty. Since warp_to_square() always outputs a fixed size, cell boundaries
are simple fixed arithmetic here — no ratio concerns at this stage.
"""

import cv2
import numpy as np

CELL_MARGIN_FRAC = 0.08   # fraction of cell size to crop off each edge
EMPTY_PIXEL_THRESH = 0.1  # fraction of non-zero pixels below which a cell
                            # is considered empty

def _detect_grid_line_positions(warped_gray, size, n=9, search_frac=0.15):
    """Detects the (n+1) grid line positions along both axes by locating
    the strongest line near each nominal (uniformly-spaced) position,
    instead of trusting uniform spacing outright. The outer perspective
    warp is only as accurate as the 4 detected corners -- small errors
    there don't distribute evenly across the grid, so nominal (arithmetic)
    cell boundaries drift off the true lines by a varying amount depending
    on position. This pulls boundaries back onto the real lines.
    Returns (y_lines, x_lines), each a list of n+1 ints."""
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    step = size / n
    search = max(3, int(step * search_frac))
    row_profile = thresh.sum(axis=1).astype(np.float64)
    col_profile = thresh.sum(axis=0).astype(np.float64)
    row_bg = np.median(row_profile)
    col_bg = np.median(col_profile)

    def _lines(profile, bg):
        lines = []
        for i in range(n + 1):
            nominal = int(round(i * step))
            if i == 0 or i == n:
                # outer border: trust the warp itself here, searching near
                # the image edge risks a partial/unreliable window
                lines.append(nominal)
                continue
            lo = max(0, nominal - search)
            hi = min(size, nominal + search + 1)
            window = profile[lo:hi]
            best_idx = int(np.argmax(window))
            best_val = window[best_idx]
            # only trust the detected peak if it's clearly a line, not
            # noise/a digit stroke -- else fall back to nominal position
            if best_val > bg * 1.5:
                lines.append(lo + best_idx)
            else:
                lines.append(nominal)
        return lines

    return _lines(row_profile, row_bg), _lines(col_profile, col_bg)

def split_into_cells(warped_gray, margin_frac=CELL_MARGIN_FRAC):
    """Split a square grayscale image into an 81-length list of cell crops,
    row-major order, using detected true grid line positions rather than
    assumed uniform spacing."""
    size = warped_gray.shape[0]
    y_lines, x_lines = _detect_grid_line_positions(warped_gray, size)

    cells = []
    for row in range(9):
        for col in range(9):
            y1, y2 = y_lines[row], y_lines[row + 1]
            x1, x2 = x_lines[col], x_lines[col + 1]
            my = int((y2 - y1) * margin_frac)
            mx = int((x2 - x1) * margin_frac)
            cell = warped_gray[y1 + my:y2 - my, x1 + mx:x2 - mx]
            cells.append(cell)
    return cells


def is_cell_empty(cell, pixel_thresh=EMPTY_PIXEL_THRESH):
    """A cell is 'empty' if very few pixels are foreground after
    thresholding. Uses Otsu since each cell crop is small and roughly
    bimodal (digit vs background)."""

    if cell.size == 0:
        return True
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    # _, thresh = cv2.threshold(blur, 0, 255,
    #                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    # print(cell.shape)
    # cv2.imshow("Cell thresh", thresh)
    # ignore a thin border ring in case grid lines leaked into the crop
    h, w = thresh.shape
    bw = max(1, int(0.08 * w))
    bh = max(1, int(0.08 * h))
    inner = thresh[bh:h - bh, bw:w - bw]
    if inner.size == 0:
        inner = thresh

    fg_fraction = np.count_nonzero(inner) / inner.size
    return fg_fraction < pixel_thresh


def extract_cells_with_empty_mask(warped_gray):
    """Convenience wrapper: returns (cells, empty_mask) where empty_mask is
    an 81-length list of booleans aligned with cells."""
    cells = split_into_cells(warped_gray)
    empty_mask = [is_cell_empty(c) for c in cells]
    return cells, empty_mask

def visualize_cells_on_warped(warped_bgr, warped_gray, cells, empty_mask,
                              margin_frac=CELL_MARGIN_FRAC,
                              alpha=0.25, text_scale=0.5, text_thickness=1,
                              save_path=None, show=True):
    """
    Draws an overlay on `warped_bgr` showing each extracted cell and whether
    it's classified empty (green) or non-empty (red).

    Parameters:
    - warped_bgr: BGR image (H,W,3) of the warped grid.
    - warped_gray: single-channel (H,W) grayscale warped image.
    - cells: list of 81 grayscale cell crops (Hc,Wc).
    - empty_mask: list of 81 booleans (True=empty).
    - margin_frac: same margin fraction used to crop cells.
    - alpha: overlay transparency (0..1).
    - save_path: optional path to save the visualization PNG.
    - show: if True, attempts to display with cv2.imshow (falls back to save only).

    Returns: annotated image (BGR numpy array).
    """
  
    # ensure color image
    vis = warped_bgr.copy()
    if vis.ndim == 2 or vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    size = warped_gray.shape[0]
    y_lines, x_lines = _detect_grid_line_positions(warped_gray, size)

    overlay = vis.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in range(81):
        row = idx // 9
        col = idx % 9
        y1, y2 = y_lines[row], y_lines[row + 1]
        x1, x2 = x_lines[col], x_lines[col + 1]

        # color: green = empty, red = digit present
        if empty_mask[idx]:
            color = (0, 255, 0)
            label = "E"
        else:
            color = (0, 0, 255)
            label = "D"

        # filled translucent box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # border
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # optional small thumbnail of the cell in the top-left corner of the cell box
        cell = cells[idx]
        if cell is not None and cell.size > 0:
            th, tw = y2 - y1, x2 - x1
            # scale cell to fit into 30% of cell box
            max_dim = max(1, int(min(th, tw) * 0.30))
            thumb = cv2.resize(cell, (max_dim, max_dim), interpolation=cv2.INTER_AREA)
            if thumb.ndim == 2:
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            else:
                thumb_bgr = thumb
            vis[y1:y1+max_dim, x1:x1+max_dim] = thumb_bgr

        # label (centered)
        txt_size = cv2.getTextSize(label, font, text_scale, text_thickness)[0]
        txt_x = x1 + (x2 - x1 - txt_size[0]) // 2
        txt_y = y1 + (y2 - y1 + txt_size[1]) // 2
        cv2.putText(vis, label, (txt_x, txt_y), font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    # blend overlay
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    if save_path:
        ok = cv2.imwrite(save_path, vis)
        if not ok:
            print(f"  [WARNING] Failed to write image to {save_path}")
    if show:
        try:
            cv2.namedWindow("Cells visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Cells visualization", 800, 600)
            cv2.imshow("Cells visualization", vis)
            cv2.waitKey(0)
            cv2.destroyWindow("Cells visualization")
        except Exception:
            # headless environment: just save (if save_path provided) or skip showing
            pass

    return vis