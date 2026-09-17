import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as V.
    """
    # Write code here
    V=np.array(V,dtype=float)
    
    delta = r+gamma*(V[s_next])-V[s]
    V[s]=V[s]+alpha*delta

    return V
    pass