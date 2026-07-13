from .solve import solve_matrix, validate_matrix, SudokuValidationError

def solve_sudoku(matrix, raise_on_invalid=True):

    solved_matrix, conflicts = solve_matrix(matrix, raise_on_invalid=raise_on_invalid)
    return solved_matrix, conflicts

def validate_sudoku(matrix):

    return validate_matrix(matrix)