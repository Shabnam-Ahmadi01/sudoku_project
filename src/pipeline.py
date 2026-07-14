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
from src.recognition.model import RandomGaussianBlur
from src.overlay import draw_solution
from src.recognition import predict_from_cells
from src.preprocessing import process_image
from config import MODEL_PATH,TEST_DATA_PATH

def sudoku_pipeline(img_path,model, debug=False, confidence_threshold=0.0):
    status = 0

    try:
        processed_image = process_image(img_path,show=debug)
    except Exception as e:
        status = 1
        if debug:
            print(f"Phase 1 failed for {img_path}: {e}")
        return None, None, None, None, status, None

    matrix,conf = predict_from_cells(model, processed_image.cells, 
                                     processed_image.empty_mask, confidence_threshold=confidence_threshold)

    solve_results = solve_sudoku(matrix, raise_on_invalid=debug)
    solved = solve_results.solved_matrix
    conflicts = solve_results.conflicts
    if(conflicts):
        status = 2
        if debug:
            print(f"Phase 3 failed for {img_path}: {conflicts}")
        return solve_results, processed_image, matrix, conf, status, None

    if(debug):
        print(conf)
        print(conflicts)

    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
        if debug:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        return solve_results, processed_image, matrix, conf, status, None
    
    result = draw_solution(original_bgr, processed_image.corners, matrix, solved)

    if(debug):
        cv2.namedWindow("Annotated result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotated result", 800, 600)
        cv2.imshow("Annotated result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return solve_results, processed_image, matrix, conf, status, result

if __name__ == "__main__":
    path = f"{TEST_DATA_PATH}/image1072.jpg"
    sudoku_pipeline(path,debug=True)