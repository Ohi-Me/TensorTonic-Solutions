import numpy as np
from math import log

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    n=len(p)
    m=len(y)
    ans=0
    for i in range(m):
        ans+=(-1)*( pow((1-p[i]),gamma)*(y[i]*log(p[i])) + pow(p[i],gamma)*((1-y[i])*log(1-p[i])) )
    return ans/m
    pass