"""
Phase 1b: Split the warped 900x900 grid into 81 cells and flag which are
empty. Since warp_to_square() always outputs a fixed size, cell boundaries
are simple fixed arithmetic here — no ratio concerns at this stage.
"""

import cv2
import numpy as np

CELL_MARGIN_FRAC = 0.16   # fraction of cell size to crop off each edge
EMPTY_PIXEL_THRESH = 0.1  # fraction of non-zero pixels below which a cell
                            # is considered empty
def _cell_boundaries(thresh, size, n=9, search_frac=0.15, margin_frac=CELL_MARGIN_FRAC):
    """For each cell, locally detects its 4 edge positions by searching
    only the pixels near that specific cell, rather than the whole grid.
    Grid lines aren't perfectly straight after the perspective warp --
    summing a full row/column smears a line's true position across many
    pixels and often loses it. A local search over just this cell's own
    row/column band stays accurate even when the line drifts slightly
    elsewhere in the image."""
    step = size / n
    search = max(3, int(step * search_frac))

    def find_edge(profile, nominal, bg):
        lo = max(0, nominal - search)
        hi = min(len(profile), nominal + search + 1)
        window = profile[lo:hi]
        if len(window) == 0:
            return nominal
        best_idx = int(np.argmax(window))
        if window[best_idx] > bg * 1.5:
            return lo + best_idx
        return nominal   # no clear line found nearby -- trust the grid math

    boundaries = []
    for row in range(n):
        for col in range(n):
            y_top_nom = int(round(row * step))
            y_bot_nom = int(round((row + 1) * step))
            x_left_nom = int(round(col * step))
            x_right_nom = int(round((col + 1) * step))

            # local band: only this cell's own column range, for finding
            # its top/bottom edges
            col_band = thresh[:, x_left_nom:x_right_nom]
            row_profile = col_band.sum(axis=1).astype(np.float64)
            row_bg = np.median(row_profile) if row_profile.size else 0

            # local band: only this cell's own row range, for finding
            # its left/right edges
            row_band = thresh[y_top_nom:y_bot_nom, :]
            col_profile = row_band.sum(axis=0).astype(np.float64)
            col_bg = np.median(col_profile) if col_profile.size else 0

            y1 = y_top_nom if row == 0 else find_edge(row_profile, y_top_nom, row_bg)
            y2 = y_bot_nom if row == n - 1 else find_edge(row_profile, y_bot_nom, row_bg)
            x1 = x_left_nom if col == 0 else find_edge(col_profile, x_left_nom, col_bg)
            x2 = x_right_nom if col == n - 1 else find_edge(col_profile, x_right_nom, col_bg)

            my = int((y2 - y1) * margin_frac)
            mx = int((x2 - x1) * margin_frac)
            boundaries.append((y1 + my, y2 - my, x1 + mx, x2 - mx))
    return boundaries


def split_into_cells(warped_gray, margin_frac=CELL_MARGIN_FRAC):
    """Split a square grayscale image into an 81-length list of cell crops,
    row-major order, using per-cell local edge detection rather than
    assumed uniform spacing."""
    size = warped_gray.shape[0]
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    boundaries = _cell_boundaries(thresh, size, margin_frac=margin_frac)

    cells = []
    for (y1, y2, x1, x2) in boundaries:
        cells.append(warped_gray[y1:y2, x1:x2])
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
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    boundaries = _cell_boundaries(thresh, size, margin_frac=margin_frac)

    overlay = vis.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in range(81):
        y1, y2, x1, x2 = boundaries[idx]

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