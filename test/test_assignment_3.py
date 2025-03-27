import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from main.assignment_3 import (
    gaussian_elimination_solve,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)

def test_gaussian_elimination():
    # Example usage
    import numpy as np
    
    A = np.array([
        [2, -1, 1],
        [1, 3, 1],
        [-1, 5, 4]
    ], dtype=float)
    b = np.array([6, 0, -3], dtype=float)
    
    x = gaussian_elimination_solve(A, b)
    print("Solution from test file:", x)

def test_lu_factorization():
    # Example usage
    import numpy as np
    
    A = np.array([
        [1,  1,  0,  3],
        [2,  1, -1,  1],
        [3, -1, -1,  2],
        [-1, 2,  3, -1]
    ], dtype=float)
    
    L, U = lu_factorization(A)
    print("L from test file:\n", L)
    print("U from test file:\n", U)

def test_is_positive_definite():
    # Example usage
    import numpy as np
    
    A = np.array([
        [9,  0,  5,  2,  1],
        [3,  9,  1,  2,  1],
        [0,  1,  7,  2,  3],
        [4,  2,  3, 12, 2],
        [3, 2,   4,   0,   8]
    ], dtype=float)
    
    pd = is_positive_definite(A)
    print("Q4")
    if pd == True: print("Pass")
    else: print("Fail")

def test_is_diagonally_dominant():
    # Example usage
    import numpy as np
    
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ], dtype=float)
    
    dd = is_diagonally_dominant(A)
    print("Q3")
    if dd == False:
        print("Pass")
    else:
        print("Fail")

if __name__ == "__main__":
    test_gaussian_elimination()
    test_lu_factorization()
    test_is_positive_definite()
    test_is_diagonally_dominant()
