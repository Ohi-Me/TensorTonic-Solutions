import numpy as np
from math import pow

def normalize_3d(v):
    v = np.asarray(v, dtype=float)
    
    # case 1: single vector (3,)
    if v.ndim == 1:
        n = len(v)
        mag = 0
        for i in range(n):
            mag += pow(v[i], 2)
        mag = pow(mag, 0.5)
        
        if mag == 0:
            return v
        
        for i in range(n):
            v[i] = v[i] / mag
        
        return v
    
    # case 2: batch (N,3)
    else:
        n = len(v)
        res = np.zeros_like(v)
        
        for i in range(n):
            mag = 0
            for j in range(3):
                mag += pow(v[i][j], 2)
            mag = pow(mag, 0.5)
            
            if mag == 0:
                res[i] = v[i]
            else:
                for j in range(3):
                    res[i][j] = v[i][j] / mag
        
        return res