import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # [N,D]=shape(X)
    # w=np.zeros(D)
    # b=0.0
    # z=[]
    # ans=0
    # for i in range(N):
    #     z=X[i]@w+b
    #     sigZ=_sigmoid(z)
    #     LoggLoss=0
    #     for j in range(D):
    #         LoggLoss=y[i]*log(sigZ) + (1-y[i])*log(1-sigZ)
    #     LoggLoss/=(-N)

    #     w=w+ lr*LoggLoss
    #     b=b+ lr*LoggLoss
    #     ans=LoggLoss
    # return ans
    N,D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        z = X@w + b
        sigZ = _sigmoid(z)

        LoggLoss = -(y*np.log(sigZ) + (1-y)*np.log(1-sigZ)).mean()

        err = sigZ - y
        dw = (np.transpose(X) @ err)/N
        db = np.mean(err)

        w = w - lr*dw
        b = b - lr*db

    return w,b
    pass