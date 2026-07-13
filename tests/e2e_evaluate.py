 
import os
import sys
import glob
import argparse
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recognition.dat_parser import parse_dat_file
from src.solver import solve_sudoku
from src.recognition import predict_matrix
from src.preprocessing import process_image,visualize_cells
from src.recognition.model import RandomGaussianBlur
from config import DATA_PATH, MODEL_PATH

import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error

# Suppress CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU detection


def matrices_equal(a, b):
    return all(a[r][c] == b[r][c] for r in range(9) for c in range(9))
 
def save_failed_attempt(process_result,output_path,status,prediction):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_cells(process_result=process_result, save_path=output_path,show=False)

    # print(f"Image saved to: {output_path}")
    txt_path = os.path.splitext(output_path)[0] + '.txt'
    
    with open(txt_path, 'w') as f:
        f.write(status)
        f.write(np.array2string(prediction))
    
    # print(f"Text file saved to: {txt_path}")


def evaluate_dataset(root_dir,model,confidence_threshold=0.6,
                      max_images=None, verbose=True):
    images = []
    for i, img_path in enumerate(glob.glob(os.path.join(root_dir, "*.jpg"))):
        if os.path.exists(img_path):
            images.append(img_path)
        if i==max_images:
            break
    
    results = {
        "total": 0,
        "grid_detection_failed": 0,
        "recognition_invalid": 0,     # phase 3 caught a conflict
        "unsolvable": 0,              # valid but no solution found
        "solved_but_wrong": 0,        # solved, but != true solution
        "hit": 0,                     # solved AND matches true solution
    }
    per_image = []
 
    for img_path in images:
        dat_path = img_path.replace(".jpg", ".dat")
        results["total"] += 1
        record = {"image": os.path.basename(img_path)}
 
        label_data = parse_dat_file(dat_path)
        true_given = label_data["matrix"]
 
        # ground truth: solve the TRUE given matrix directly (bypasses
        # recognition entirely) -- this is "the right answer"
        try:
            true_solution, _ = solve_sudoku(true_given, raise_on_invalid=False)
        except ValueError:
            true_solution = None
        if true_solution is None:
            # shouldn't happen for a well-formed dataset, but skip rather
            # than crash if a .dat file itself is broken
            if verbose:
                print(f"  [skip] {record['image']}: ground truth unsolvable, "
                      f"check .dat file")
            results["total"] -= 1
            continue
 
        try:
            process_res = process_image(img_path)
        except Exception as e:
            results["grid_detection_failed"] += 1
            record["outcome"] = "grid_detection_failed"
            per_image.append(record)
            if verbose:
                print(f"  [FAIL grid] {record['image']}: {e}")
            continue
 
        recognized_matrix, confidences = predict_matrix(
            img_path=img_path,
            model=model,
            confidence_threshold=confidence_threshold,
        )
        record["recognized_matrix"] = recognized_matrix
        record["mean_confidence"] = float(confidences.mean())
 
        try:
            solved, conflicts = solve_sudoku(recognized_matrix, raise_on_invalid=False)
        except ValueError:
            solved, conflicts = None, []

        filename = os.path.basename(img_path)
        if conflicts:
            results["recognition_invalid"] += 1
            record["outcome"] = "recognition_invalid"
            record["conflicts"] = conflicts
            save_failed_attempt(process_res,output_path=f"../outputs/failed/{filename}",
                                prediction=recognized_matrix, status="recognition_invalid")
        elif solved is None:
            results["unsolvable"] += 1
            record["outcome"] = "unsolvable"
            save_failed_attempt(process_res,output_path=f"../outputs/failed/{filename}",
                                prediction=recognized_matrix, status="unsolvable")
        elif matrices_equal(solved, true_solution):
            results["hit"] += 1
            record["outcome"] = "hit"
        else:
            save_failed_attempt(process_res,output_path=f"../outputs/failed/{filename}",
                                prediction=recognized_matrix,status="solved_but_wrong")
            results["solved_but_wrong"] += 1
            record["outcome"] = "solved_but_wrong"
 
        if verbose:
            print(f"  [{record['outcome']}] {record['image']} "
                  f"(mean conf: {record['mean_confidence']:.2f})")
 
        per_image.append(record)
 
    return results, per_image
 
 
def print_summary(results):
    total = results["total"]
    if total == 0:
        print("No valid image/.dat pairs found.")
        return
 
    print("\n" + "=" * 50)
    print(f"End-to-end results over {total} puzzles")
    print("=" * 50)
    print(f"  Hit (fully correct):        {results['hit']:4d}  "
          f"({100 * results['hit'] / total:.1f}%)")
    print(f"  Solved but wrong:           {results['solved_but_wrong']:4d}  "
          f"({100 * results['solved_but_wrong'] / total:.1f}%)")
    print(f"  Recognition invalid (P3):   {results['recognition_invalid']:4d}  "
          f"({100 * results['recognition_invalid'] / total:.1f}%)")
    print(f"  Unsolvable (valid input):   {results['unsolvable']:4d}  "
          f"({100 * results['unsolvable'] / total:.1f}%)")
    print(f"  Grid detection failed (P1): {results['grid_detection_failed']:4d}  "
          f"({100 * results['grid_detection_failed'] / total:.1f}%)")
    print("=" * 50)
    print(f"  END-TO-END ACCURACY: {100 * results['hit'] / total:.1f}%")
    print("=" * 50)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="path to v2_train or v2_test folder",default="../data/v2_train/v2_train")
    parser.add_argument("--model", default="../models/mnist_cnn.keras")
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
 
    from tensorflow import keras
    model = keras.models.load_model(args.model)
 
    results, per_image = evaluate_dataset(
        model=model,
        root_dir=args.data_root,
        confidence_threshold=args.confidence_threshold,
        max_images=args.max_images,
    )
    print_summary(results)
 
 
if __name__ == "__main__":
    main()

