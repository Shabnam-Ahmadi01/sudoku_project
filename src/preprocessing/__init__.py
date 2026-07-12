import cv2
import matplotlib.pyplot as plt

from schemas import ProcessResult
from .grid_detector import detect_and_warp
from .cell_extractor import extract_cells_with_empty_mask,visualize_cells_on_warped

def show_image(title, img):
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 600)
        if img.ndim == 2:
            cv2.imshow(title, img)
        else:
            cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    except Exception:
        plt.figure(figsize=(6,6))
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

def visualize_cells(process_result, save_path=None, show="False"):
    visualize_cells_on_warped(process_result.warped_bgr, process_result.warped_gray, 
                              process_result.cells, process_result.empty_mask,
                                    show=show, save_path=save_path)
    
def process_image(path, show=False):
    """Returns a dict:
        warped_bgr   - 900x900 color image of the rectified grid
        warped_gray  - 900x900 grayscale version
        cells        - list of 81 grayscale cell crops, row-major
        empty_mask   - list of 81 booleans, True if cell looks empty
    Raises ValueError if no grid could be detected.
    """
    warped_bgr, warped_gray, corners, thresh = detect_and_warp(path, debug=True)
    cells, empty_mask = extract_cells_with_empty_mask(warped_gray)

    result = ProcessResult(warped_bgr, warped_gray, cells, empty_mask, corners)

    if show:
        show_image("Warped grid", warped_gray)
        visualize_cells(result,show=True)
        
    return result