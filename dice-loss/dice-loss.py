import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    # p=np.asarray(p)
    # y=np.asarray(y)
    p=np.asarray(p).flatten()
    y=np.asarray(y).flatten()
    n=len(p)
    pSum=0
    ySum=0
    curr=0
    for i in range(n):
        curr+=p[i]*y[i]
        pSum+=p[i]
        ySum+=y[i]
    curr=2*curr+eps
    ans=curr/(pSum+ySum+eps)
    return 1-ans
    pass