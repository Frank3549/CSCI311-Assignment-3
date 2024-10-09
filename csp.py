"""
CS311 Programming Assignment 3: CSP

Full Name: Frank Bautista

Brief description of my solver:

TODO Briefly describe your solver. Did it perform better than the AC3 solver? If so, why do you think so? If not, can you think of any ways to improve it?
"""

import argparse, time
from functools import wraps
from typing import Dict, List, Optional, Set, Tuple

# You are welcome to add constants, but do not modify the pre-existing constants

# Length of side of a Soduku board
SIDE = 9

# Length of side of "box" within a Soduku board
BOX = 3

# Domain for cells in Soduku board
DOMAIN = range(1, 10)

# Helper constant for checking a Soduku solution
SOLUTION = set(DOMAIN)

# Own constants
top_left_grid = {0, 1, 2, 9, 10, 11, 18, 19, 20}
top_middle_grid = {3, 4, 5, 12, 13, 14, 21, 22, 23}
top_right_grid = {6, 7, 8, 15, 16, 17, 24, 25, 26}
middle_left_grid = {27, 28, 29, 36, 37, 38, 45, 46, 47}
middle_middle_grid = {30, 31, 32, 39, 40, 41, 48, 49, 50}
middle_right_grid = {33, 34, 35, 42, 43, 44, 51, 52, 53}
bottom_left_grid = {54, 55, 56, 63, 64, 65, 72, 73, 74}
bottom_middle_grid = {57, 58, 59, 66, 67, 68, 75, 76, 77}
bottom_right_grid = {60, 61, 62, 69, 70, 71, 78, 79, 80}

grids = [top_left_grid, top_middle_grid, top_right_grid, 
         middle_left_grid, middle_middle_grid, middle_right_grid, 
         bottom_left_grid, bottom_middle_grid, bottom_right_grid
        ]







def check_solution(board: List[int], original_board: List[int]) -> bool:
    """Return True if board is a valid Sudoku solution to original_board puzzle"""
    # Original board values are maintained
    for s, o in zip(board, original_board):
        if o != 0 and s != o:
            return False
    for i in range(SIDE):
        # Valid row
        if set(board[i * SIDE : (i + 1) * SIDE]) != SOLUTION:
            return False
        # Valid column
        if set(board[i : SIDE * SIDE : SIDE]) != SOLUTION:
            return False
        # Valid Box (here i is serving as the "box" id since there are SIDE boxes)
        box_row, box_col = (i // BOX) * BOX, (i % BOX) * BOX
        box = set()
        for r in range(box_row, box_row + BOX):
            box.update(board[r * SIDE + box_col : r * SIDE + box_col + BOX])
        if box != SOLUTION:
            return False
    return True


def countcalls(func):
    """Decorator to track the number of times a function is called. Provides `calls` attribute."""
    countcalls.calls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        initial_calls = countcalls.calls
        countcalls.calls += 1
        result = func(*args, **kwargs)
        wrapper.calls = countcalls.calls - initial_calls
        return result

    return wrapper


def get_all_neighbors() -> List[List[int]]:
    """
    Get the neighbors of a space in the sudoku board
    
    Args:
        index (int): Index of the space in the sudoku board
    
    Returns:
        List[int]: List of indexes of the neighbors of the space
    """
    all_neighbors: List[List[int]] = []
    for board_space_index in range(SIDE**2):

        neighbors: Set[int] = set()
        row = board_space_index // SIDE
        col = board_space_index % SIDE
        neighbors.update(range(row*SIDE, row*SIDE + SIDE)) # add row neighbors
        neighbors.update(range(col, SIDE**2, SIDE)) # add col neighbors
        for grid in grids: #grid neighbors
            if board_space_index in grid:
                neighbors.update(grid) 
                break
        neighbors.remove(board_space_index) #remove self
        all_neighbors.append(list(neighbors))

    return all_neighbors


def initialize_queue(neighbors: List[List[int]]) -> Set[Tuple[int, int]]:
    queue = set()
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            queue.add((i, j))
    return queue
    
def print_board(board: List[int]) -> None:
    for i in range(SIDE):
        print(board[i*SIDE:(i+1)*SIDE])
    print()
    
        
            
# The @countcalls decorator tracks the number of times we call the recursive function. Make sure the decorator
# is included on your recursive search function if you change the implementation.
@countcalls
def backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """
    # TODO: Implement the backtracking search algorithm here
    return None




def sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using backtracking search with the AC3 algorithm

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    domains = [[val] if val else list(DOMAIN) for val in board] # list of domains for each variable
    neighbors = get_all_neighbors()
    assignment = {}
    queue = set()

    # TODO: Complete the initialization of the neighbors and queue data structures

    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    # While not necessary for correctness, initializing the assignment improves performance, especially
    # for plain backtracking search.
    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }
    result = backtracking_search(neighbors, queue, domains, assignment)

    # Convert result dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, backtracking_search.calls




@countcalls
def my_backtracking_search(
    neighbors: List[List[int]],
    queue: Set[Tuple[int, int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
) -> Optional[Dict[int, int]]:
    """Custom backtracking search implementing efficient heuristics

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable
        assignment (Dict[int, int]): Current variable->value assignment 

    Returns:
        Optional[Dict[int, int]]: Solution or None indicating no solution found
    """
    return None




def my_sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using your own custom solver

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and
            a count of calls to recursive backtracking function
    """

    domains = [[val] if val else list(DOMAIN) for val in board]
    neighbors = []
    queue = set()

    # TODO: Complete the initialization of the neighbors and queue data structures

    # Initialize the assignment for any squares with domains of size 1 (e.g., pre-specified squares).
    assignment = {
        var: domain[0] for var, domain in enumerate(domains) if len(domain) == 1
    }

    result = my_backtracking_search(neighbors, queue, domains, assignment)

    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, my_backtracking_search.calls


if __name__ == "__main__":
    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(description="Run sudoku solver")
    parser.add_argument(
        "-a",
        "--algo",
        default="ac3",
        help="Algorithm (one of ac3, custom)",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="easy",
        help="Difficulty level (one of easy, medium, hard)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        type=int,
        help="Number of trials for timing",
    )
    parser.add_argument("puzzle", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # fmt: off
    if args.puzzle:
        board = [int(c) for c in args.puzzle]
        if len(board) != SIDE*SIDE or set(board) > (set(DOMAIN) | { 0 }):
            raise ValueError("Invalid puzzle specification, it must be board length string with digits 0-9")
    elif args.level == "easy":
        board = [
            0,0,0,1,3,0,0,0,0,
            7,0,0,0,4,2,0,8,3,
            8,0,0,0,0,0,0,4,0,
            0,6,0,0,8,4,0,3,9,
            0,0,0,0,0,0,0,0,0,
            9,8,0,3,6,0,0,5,0,
            0,1,0,0,0,0,0,0,4,
            3,4,0,5,2,0,0,0,8,
            0,0,0,0,7,3,0,0,0,
        ]
    elif args.level == "medium":
        board = [
            0,4,0,0,9,8,0,0,5,
            0,0,0,4,0,0,6,0,8,
            0,5,0,0,0,0,0,0,0,
            7,0,1,0,0,9,0,2,0,
            0,0,0,0,8,0,0,0,0,
            0,9,0,6,0,0,3,0,1,
            0,0,0,0,0,0,0,7,0,
            6,0,2,0,0,7,0,0,0,
            3,0,0,8,4,0,0,6,0,
        ]
    elif args.level == "hard":
        board = [
            1,2,0,4,0,0,3,0,0,
            3,0,0,0,1,0,0,5,0,  
            0,0,6,0,0,0,1,0,0,  
            7,0,0,0,9,0,0,0,0,    
            0,4,0,6,0,3,0,0,0,    
            0,0,3,0,0,2,0,0,0,    
            5,0,0,0,8,0,7,0,0,    
            0,0,7,0,0,0,0,0,5,    
            0,0,0,0,0,0,0,9,8,
        ]
    else:
        raise ValueError("Unknown level")
    # fmt: on

    if args.algo == "ac3":
        solver = sudoku
    elif args.algo == "custom":
        solver = my_sudoku
    else:
        raise ValueError("Unknown algorithm type")

    times = []
    for i in range(args.trials):
        test_board = board[:]  # Ensure original board is not modified
        start = time.perf_counter()
        solution, recursions = solver(test_board)
        end = time.perf_counter()
        times.append(end - start)
        if solution and not check_solution(solution, board):
            print(solution)
            raise ValueError("Invalid solution")

        if solution:
            print(f"Trial {i} solved with {recursions} recursions")
            print(solution)
        else:
            print(f"Trial {i} not solved with {recursions} recursions")

    print(
        f"Minimum time {min(times)}s, Average time {sum(times) / args.trials}s (over {args.trials} trials)"
    )
