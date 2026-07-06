"""
Phase 4a: Solve a recognized 9x9 sudoku matrix using the open-source
py-sudoku library (pip: py-sudoku, import name: sudoku).

Includes constraint validation (duplicate digits in a row/col/box) BEFORE
attempting to solve -- this is the safety net for Phase 3 recognition
errors. A board can be structurally invalid (duplicate given digits, which
means the CNN misread something) or structurally valid but unsolvable
(rare, but possible with enough misreads); we distinguish the two so the
caller can react appropriately (e.g. "re-check this cell" vs "no solution
exists").
"""

from sudoku import Sudoku


class SudokuValidationError(Exception):
    """Raised when the input matrix violates sudoku constraints (i.e. the
    recognized digits themselves conflict -- almost always a Phase 2/3
    recognition error, not a puzzle issue)."""
    def __init__(self, message, conflicts):
        super().__init__(message)
        self.conflicts = conflicts   # list of (row, col, reason) tuples


def _iter_units():
    """Yields lists of (row, col) coordinates for every row, column, and
    3x3 box -- the 27 constraint groups in a sudoku."""
    for r in range(9):
        yield [(r, c) for c in range(9)]
    for c in range(9):
        yield [(r, c) for r in range(9)]
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            yield [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]


def validate_matrix(matrix):
    """Checks for duplicate non-zero digits within any row/col/3x3 box.
    Returns a list of conflicts: each is (row, col, value, unit_type).
    Empty list means the board is internally consistent (still doesn't
    guarantee a solution exists, just that there's no immediate
    contradiction)."""
    if len(matrix) != 9 or any(len(row) != 9 for row in matrix):
        raise ValueError("matrix must be 9x9")

    conflicts = []
    for unit in _iter_units():
        seen = {}
        for (r, c) in unit:
            v = matrix[r][c]
            if v == 0:
                continue
            if not (1 <= v <= 9):
                conflicts.append((r, c, v, "out_of_range"))
                continue
            if v in seen:
                conflicts.append((r, c, v, "duplicate"))
                conflicts.append((*seen[v], v, "duplicate"))
            else:
                seen[v] = (r, c)
    # de-duplicate (a cell can be flagged twice if it participates in two
    # conflicting units)
    return sorted(set(conflicts))


def solve_matrix(matrix, raise_on_invalid=True):
    """Validates then solves a 9x9 matrix (0 = empty).
    Returns (solved_matrix, conflicts) where:
      - if conflicts is non-empty, solved_matrix is None (board is
        internally inconsistent, almost certainly a recognition error)
      - if the board is consistent but has no solution, raises ValueError
      - otherwise returns the fully solved 9x9 matrix
    """
    conflicts = validate_matrix(matrix)
    if conflicts:
        if raise_on_invalid:
            raise SudokuValidationError(
                f"Recognized board has {len(conflicts)} constraint "
                f"conflict(s) -- likely a digit recognition error.",
                conflicts,
            )
        return None, conflicts

    board_none = [[None if v == 0 else v for v in row] for row in matrix]
    puzzle = Sudoku(3, 3, board=board_none)

    try:
        solution = puzzle.solve(assert_solvable=True)
    except Exception as e:
        raise ValueError(f"Board is consistent but has no solution: {e}")

    return solution.board, []