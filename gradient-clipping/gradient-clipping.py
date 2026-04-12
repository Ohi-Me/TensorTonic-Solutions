import numpy as np
from math import sqrt

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g=np.asarray(g, dtype=float)
    # g_mag = sum(x*x for x in g)
    # g_mag = sqrt(g_mag)
    g_mag=sqrt(np.sum(g*g))

    if max_norm<=0:
        return g

    if g_mag>max_norm:
        for i in range(len(g)):
            g[i] = g[i] * (max_norm / g_mag)
    return g
    pass