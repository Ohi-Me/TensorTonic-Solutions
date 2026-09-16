import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    N=len(y_pred)
    L=0
    
    for i in range(len(y_true)):
        L+=(-np.log(y_pred[i][y_true[i]]))

    L/=N
    return L
    pass