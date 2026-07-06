from .render import draw_solution_on_original

def draw_solution(original_bgr, corners, given_matrix,
                               solved_matrix, color=(40, 200, 40)):
    return draw_solution_on_original(original_bgr, corners, given_matrix, solved_matrix, color)
