# schemas.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ProcessResult:
    warped_bgr: np.ndarray
    warped_gray: np.ndarray
    cells: list          # 81 grayscale crops, row-major
    empty_mask: list      # 81 bools
    corners: np.ndarray   # 4x2, original image coords

@dataclass
class SolveResult:
    solved_matrix: list | None   # 9x9 ints, or None if invalid
    conflicts: list              # (row, col, value, reason) tuples