from .solve import solve_matrix, validate_matrix, SudokuValidationError
from schemas import SolveResult

def solve_sudoku(matrix, raise_on_invalid=True):

    solved_matrix, conflicts = solve_matrix(matrix, raise_on_invalid=raise_on_invalid)
    return SolveResult(solved_matrix=solved_matrix, conflicts=conflicts)

def validate_sudoku(matrix):

    return validate_matrix(matrix)