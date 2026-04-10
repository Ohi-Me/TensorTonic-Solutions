import numpy as np
from math import pow,sqrt

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    # x = np.asarray(x, dtype=float)  
    # gamma = np.asarray(gamma, dtype=float)
    # beta = np.asarray(beta, dtype=float)
    # n=len(x)
    # miu=0
    # for i in range(n):
    #     miu+=x[i]
    # miu/=n

    # sig=0
    # for i in range(n):
    #     sig=(x[i]-miu)**2
    # sig/=n

    # x_new=[]
    # for i in range(n):
    #     x_new[i]=(x[i]-miu)/sqrt((sig+eps))

    # y_new=[]
    # for i in range(n):
    #     y_new=gamma*x_new[i]+beta

    # return y_new

    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        # (N, D)
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_hat = (x - mu) / np.sqrt(var + eps)
        y = gamma * x_hat + beta

    else:
        # (N, C, H, W)
        mu = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.var(x, axis=(0,2,3), keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)

        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        y = gamma * x_hat + beta

    return y
    pass