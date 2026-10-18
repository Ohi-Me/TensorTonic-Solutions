import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x=np.array(x, dtype=float)

    if x.ndim==1:
        x=x-np.max(x)
        exp_x=np.exp(x)
        return exp_x/np.sum(exp_x)

    result=[]
    for row in x:
        row=row-np.max(row)
        exp_x=np.exp(row)
        result.append(exp_x/np.sum(exp_x))

    return np.array(result)