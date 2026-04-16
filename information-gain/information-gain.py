import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y=np.asarray(y,dtype=float)
    HL=[]
    HR=[]
    n=len(y)
    nL=0
    nR=0

    for i in range(n):
        if(split_mask[i]==True):
            HL.append(y[i])
            nL+=1
        else:
            HR.append(y[i])
            nR+=1

    return (_entropy(y))-((nL/n)*_entropy(HL) + (nR/n)*_entropy(HR))
    pass
