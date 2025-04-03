import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from main.assignment_3 import (
    gaussian_elimination_solve,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite,
)

def test_gaussian_elimination():
    A = np.array([
        [2, -1, 1],
        [1, 3, 1],
        [-1, 5, 4]
    ], dtype=float)
    b = np.array([6, 0, -3], dtype=float)
    
    x = gaussian_elimination_solve(A, b)
    x.tolist()
    sol = np.array([2, -1,  1])
    if np.allclose(x, sol): print("Pass")
    else: print("Fail")
    # print(x)
    

def test_lu_factorization():
    A = np.array([
        [1,  1,  0,  3],
        [2,  1, -1,  1],
        [3, -1, -1,  2],
        [-1, 2,  3, -1]
    ], dtype=float)
    
    L, U = lu_factorization(A)
    # print("L from test file:\n", L)
    # print("U from test file:\n", U)
    solL, solU = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [3, 4, 1, 0], [-1, -3, 0, 1]]), np.array([[1, 1, 0, 3], [0, -1, -1, -5], [0, 0, 3, 13], [0, 0, 0, -13]])
    if np.allclose(L, solL) and np.allclose(U, solU): print("Pass")
    else: print("Fail")

def test_is_diagonally_dominant():
    A = np.array([
        [9,  0,  5,  2,  1],
        [3,  9,  1,  2,  1],
        [0,  1,  7,  2,  3],
        [4,  2,  3, 12, 2],
        [3, 2,   4,   0,   8]
    ], dtype=float)
    
    dd = is_diagonally_dominant(A)
    
    if dd == False: 
        print("Pass")
    else: 
        print("Fail")

def test_is_positive_definite():
    A = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ], dtype=float)
    
    pd = is_positive_definite(A)

    if pd == True:
        print("Pass")
    else:
        print("Fail")

if __name__ == "__main__":
    print("Q1")
    test_gaussian_elimination()
    print("Q2")
    test_lu_factorization()
    print("Q3")
    test_is_diagonally_dominant()
    print("Q4")
    test_is_positive_definite()
