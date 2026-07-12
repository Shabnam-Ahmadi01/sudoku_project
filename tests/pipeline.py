import cv2
import numpy as np
import sys
import os
import keras
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error

# Suppress CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU detection

from src.solver import SudokuValidationError, solve_sudoku
from src.overlay import draw_solution
from src.recognition import predict_matrix
from src.preprocessing import process_image
from config import DATA_PATH,TEST_DATA_PATH,MODEL_PATH

def main():
    
    path = f"{DATA_PATH}/image1014.jpg"
    
    processed_image = process_image(path,show=True)
    print("Phase 1 OK. Corners:\n", processed_image.corners)

    model = keras.models.load_model(MODEL_PATH)
    recognized_matrix,_ = predict_matrix(model,path)
    print("Phase 2 OK. Recognized matrix:")
    for row in recognized_matrix:
        print(row)

    solved, conflicts = solve_sudoku(recognized_matrix)
    if conflicts:
        print("VALIDATION FAILED:", conflicts)
        return
    print("Solved matrix!")

    original_bgr = cv2.imread(path)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    
    result = draw_solution(original_bgr, processed_image.corners, recognized_matrix, solved)
    cv2.imwrite("outputs/e2e_result.png", result)
    print("Saved annotated result to e2e_result.png")
    cv2.namedWindow("Annotated result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotated result", 800, 600)
    cv2.imshow("Annotated result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # also sanity-check: overlay drawn on the WARPED image directly (no
    # inverse-warp) to isolate whether errors are in rendering vs in the
    # inverse-perspective step
    from overlay.render import render_overlay_square
    overlay_sq = render_overlay_square(recognized_matrix, solved)
    alpha = overlay_sq[:, :, 3:4].astype(np.float32) / 255.0
    warped_check = (overlay_sq[:, :, :3].astype(np.float32) * alpha +
                    processed_image.warped_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    cv2.imwrite("outputs/e2e_warped_check.png", warped_check)

    # Test the invalid-board path too
    bad_matrix = [row[:] for row in recognized_matrix]  # copy
    bad_matrix[0][2] = 5  # duplicate 5 in row 0 (already has a 5 at col 0)
    try:
        solve_sudoku(bad_matrix, raise_on_invalid=True)
        print("ERROR: should have raised on invalid board")
    except SudokuValidationError as e:
        print(f"\nCorrectly caught invalid board: {e}")
        print("Conflicts:", e.conflicts[:4], "...")


if __name__ == "__main__":
    main()