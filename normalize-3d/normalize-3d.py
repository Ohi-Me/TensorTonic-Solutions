import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v,dtype=float)
    n=len(v)
    mag=0

    if(v.ndim==1):
        for i in range(3):
            mag=mag + pow(v[i],2)
        if(mag==0):
            return v
        else:
            mag=pow(mag,0.5)
            for i in range(3):
                v[i]= v[i]/mag
            return v
    else:
        for i in range(n):
            mag=0
            for j in range(3):
                mag=mag + pow(v[i][j],2)
            if(mag==0):
                v[i]=v[i]
            else:
                mag= pow(mag,0.5)
                for k in range(3):
                    v[i][k]=v[i][k] / mag
    return v
    pass