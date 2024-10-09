import sys, unittest

import csp

# fmt: off
EASY = [
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
 
MEDIUM = [
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

HARD = [
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

MULTIPLE = [
    0,8,0,0,0,9,7,4,3,
    0,5,0,0,0,8,0,1,0,
    0,1,0,0,0,0,0,0,0,
    8,0,0,0,0,5,0,0,0,
    0,0,0,8,0,4,0,0,0,
    0,0,0,3,0,0,0,0,6,
    0,0,0,0,0,0,0,7,0,
    0,3,0,5,0,0,0,8,0,
    9,7,2,4,0,0,0,5,0,
]

UNSOLVABLE = [
    2,5,6,0,4,9,8,3,7,
    1,8,3,5,0,7,9,6,4,
    9,7,4,3,8,6,2,5,1,
    8,4,9,1,6,2,3,7,5,
    5,6,2,7,9,3,4,1,8,
    7,3,1,4,5,8,6,2,9,
    6,9,7,8,3,1,5,4,2,
    4,2,8,6,7,5,1,9,3,
    3,1,5,9,2,4,7,8,6,
]

UNSOLVABLE_HARD = [
    0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,2,0,0,
    0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,
]
# fmt: on

class SodukuAC3Test(unittest.TestCase):
    def test_solves_known_puzzles(self):
        # To speed up the tests, consider starting with just the EASY board, e.g.,
        # for board_name in ("EASY",):
        for board_name in ("EASY", "MEDIUM", "HARD"):
            with self.subTest(msg=f"{board_name} board"):
                board = globals()[board_name]
                solution, recursions = csp.sudoku(board[:])
                self.assertTrue(solution and csp.check_solution(solution, board))
                self.assertGreaterEqual(recursions, sum(v == 0 for v in board))

    def test_solves_multiple_solution_puzzles(self):
        solution, _ = csp.sudoku(MULTIPLE[:])
        self.assertTrue(solution and csp.check_solution(solution, MULTIPLE))

    def test_returns_failure_for_unsolvable_puzzle(self):
        solution, _ = csp.sudoku(UNSOLVABLE[:])
        self.assertIsNone(solution)

class SodukuCustomTest(unittest.TestCase):
    def test_solves_known_puzzles(self):
        # To speed up the tests, consider starting with just the EASY board, e.g., ("EASY",)
        for board_name in ("EASY", "MEDIUM", "HARD"):
            with self.subTest(msg=f"{board_name} board"):
                board = globals()[board_name]
                solution, _ = csp.my_sudoku(board[:])
                self.assertTrue(solution and csp.check_solution(solution, board))

    def test_solves_multiple_solution_puzzles(self):
        solution, _ = csp.my_sudoku(MULTIPLE[:])
        self.assertTrue(solution and csp.check_solution(solution, MULTIPLE))

    def test_returns_failure_for_unsolvable_puzzle(self):
        # When your performance has improved, try also testing with UNSOLVABLE_HARD, e.g.
        # for board_name in ("UNSOLVABLE","UNSOLVABLE_HARD"):
        for board_name in ("UNSOLVABLE",):
             with self.subTest(msg=f"{board_name} board"):
                board = globals()[board_name]
                solution, _ = csp.my_sudoku(board[:])
                self.assertIsNone(solution)

    

if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
