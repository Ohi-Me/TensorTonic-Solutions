import numpy as np
from math import sin,cos,pow

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    ans=[]
    for i in range(seq_len):
        curr=[]
        for j in range(d_model):
            if j%2==0 :
                x=sin(i/(pow(base,((2*(j//2))/d_model)))) 
            else:
                x=cos(i/(pow(base,((2*(j//2))/d_model))))
            curr.append(x)
        ans.append(curr) 
    ans=np.asarray(ans)
    return ans
    pass