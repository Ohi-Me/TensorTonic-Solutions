import numpy as np

def maxpool_forward(X, pool_size, stride):
    
    X=np.asarray(X)
    h,w = X.shape
    
    Hout = (h - pool_size)//stride + 1
    Wout = (w - pool_size)//stride + 1

    out = [[0]*Wout for _ in range(Hout)]

    for i in range(Hout):
        for j in range(Wout):
            window = X[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
            out[i][j] = np.max(window)
            
    return np.array(out).tolist()