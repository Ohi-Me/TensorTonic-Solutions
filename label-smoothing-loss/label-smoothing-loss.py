import numpy as np
from math import log,pow

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # Write code here
    K=len(predictions)
    q=[]
    for i in range(K):
        if i==target:
            x=(1-epsilon)+epsilon/K
        else:
            x=epsilon/K
        q.append(x)
    loss=0.0
    for i in range(K):
        loss-=q[i]*log(predictions[i])
    return loss
    pass