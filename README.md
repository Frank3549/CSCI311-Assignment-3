# CS311 Programming Assignment 3: CSP

For this assignment, you will be solving Sudoku puzzle as a Constraint Satisfaction Problem (CSP). Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code benchmarks your search functions on a set of game boards. Executing `csp.py` will run backtracking search with AC3 on an "easy" board by default.  You can change the algorithm, the difficulty level, the number of trials and specify the board by changing the optional arguments shown below.

```
$ python3 csp.py -h
usage: csp.py [-h] [-a ALGO] [-l LEVEL] [-t TRIALS] [puzzle]

Run sudoku solver

positional arguments:
  puzzle

optional arguments:
  -h, --help            show this help message and exit
  -a ALGO, --algo ALGO  Algorithm (one of ac3, custom)
  -l LEVEL, --level LEVEL
                        Difficulty level (one of easy, medium, hard)
  -t TRIALS, --trials TRIALS
                        Number of trials for timing
```

For example, to test your custom solver, run the program as `python3 csp.py -a custom` or to test a specific input, `python3 csp.py 003000600900305000001806400008102900700000008006708200002609500800203009005010300`.

If you are working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run csp.py -a custom`.

## Unit testing

To assist you during development, a unit test suite is provided in `csp_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `csp_test.py` file as a program, e.g. `python3 csp_test.py`. To speedup testing you might want to comment out the harder boards at first.

```
$ python3 csp_test.py
......
----------------------------------------------------------------------
Ran 6 tests in 4.167s

OK
```