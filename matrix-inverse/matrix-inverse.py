import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A=np.asarray(A,dtype=float)
    # check 2D and square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    # check singular
    if np.linalg.matrix_rank(A) < A.shape[0]:
        return None
    inv = np.linalg.inv(A)
    # validate result
    if np.linalg.norm(A @ inv - np.eye(A.shape[0])) >= 1e-7:
        return None
    return inv
    pass
