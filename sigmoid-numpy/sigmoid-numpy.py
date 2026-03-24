import numpy as np
# from math import e

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    # return 1/(1+np.exp(-x))  //this cant be used -x as exp in np array :- ❌ Python cannot do -x on a list
    # return np.exp(x)/(1+np.exp(x))  //for large x=1000 or more it will overflow
    x=np.asarray(x); #converted list or tuple into np array so that -x operation can be done
    return 1/(1+np.exp(-x))
    pass