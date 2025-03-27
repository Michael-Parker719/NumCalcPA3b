import numpy as np

def gaussian_elimination_solve(A_in, b_in):
    """
    Solve A x = b using Gaussian elimination (no partial pivoting) 
    followed by backward substitution.
    A_in and b_in are not modified in-place.
    Returns the solution vector x.
    """
    # Make copies so we don't overwrite the originals
    A = A_in.astype(float).copy()
    b = b_in.astype(float).copy()

    n = A.shape[0]

    # Forward elimination
    for i in range(n):
        # Pivot element is A[i, i] (no partial pivoting in this example)
        for j in range(i+1, n):
            if A[i, i] == 0:
                raise ValueError("Encountered zero pivot - pivoting needed or system is singular.")
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # Backward substitution
    x = np.zeros(n)
    for i in reversed(range(n)):
        # sum_{k=i+1 to n-1} A[i, k] * x[k]
        sum_ax = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i, i]

    return x


def lu_factorization(A_in):
    """
    Perform an LU factorization of matrix A_in (no pivoting).
    Returns (L, U) such that A_in = L * U.
    L is unit lower-triangular (1s on diagonal), U is upper-triangular.
    """
    A = A_in.astype(float).copy()
    n = A.shape[0]

    L = np.eye(n, dtype=float)
    U = A.copy()

    for k in range(n-1):
        # The pivot is U[k, k]
        if abs(U[k, k]) < 1e-15:
            raise ValueError("Zero pivot encountered. Pivoting needed or matrix is singular.")
        for i in range(k+1, n):
            # Multiplier
            L[i, k] = U[i, k] / U[k, k]
            # Eliminate below pivot
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return (L, U)


def is_diagonally_dominant(A_in):
    """
    Check if the given square matrix A_in is (strictly) diagonally dominant.
    That is, for every row i, |A[i,i]| > sum of the absolute values of the other elements in that row.
    Returns True/False.
    """
    A = A_in.astype(float)
    n = A.shape[0]
    for i in range(n):
        # Diagonal element
        diag = abs(A[i, i])
        # Sum of other elements in row i
        off_sum = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_sum:
            return False
    return True


def is_positive_definite(A_in):
    """
    Check if a symmetric matrix A_in is positive definite.
    One quick test: all leading principal minors > 0.
    Returns True/False.
    """
    A = A_in.astype(float)
    n = A.shape[0]
    # We can check the determinants of the leading principal submatrices
    for k in range(1, n+1):
        # principal submatrix of size k
        principal_sub = A[:k, :k]
        if np.linalg.det(principal_sub) <= 1e-14:
            return False
    return True


def main():

    A1 = np.array([
        [ 2, -1,  1],
        [ 1,  3,  1],
        [-1,  5,  4]
    ], dtype=float)
    b1 = np.array([6, 0, -3], dtype=float)

    sol1 = gaussian_elimination_solve(A1, b1)
    print("Q1:", sol1)

    A2 = np.array([
        [ 1,  1,  0,  3],
        [ 2,  1, -1,  1],
        [ 3, -1, -1,  2],
        [-1,  2,  3, -1]
    ], dtype=float)

    L, U = lu_factorization(A2)
    det_A2 = np.prod(np.diag(U))

    print("\nQ2 Determinant:", det_A2)
    print("Q2 L Matrix: \n", L)
    print("Q2 U Matrix:\n", U)

    A3 = np.array([
        [ 9,  0,  5,  2,  1],
        [ 3,  9,  1,  2,  1],
        [ 0,  1,  7,  2,  3],
        [ 4,  2,  3, 12,  2],
        [ 3,  2,  4,  0,  8]
    ], dtype=float)

    dd = is_diagonally_dominant(A3)
    print("\nQ3:", dd)

    A4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    pd = is_positive_definite(A4)
    print("\nQ4:", pd)


if __name__ == "__main__":
    main()
