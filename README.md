## Running the program

Executing `csp.py` will run backtracking search with AC3 on an "easy" board by default.  You can change the algorithm, the difficulty level, the number of trials and specify the board by changing the optional arguments shown below.

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

To test the custom solver, run the program as `python3 csp.py -a custom` or to test a specific input, `python3 csp.py 003000600900305000001806400008102900700000008006708200002609500800203009005010300`.


## Unit testing

```
$ python3 csp_test.py
......
----------------------------------------------------------------------
Ran 6 tests in 4.167s

OK
```
