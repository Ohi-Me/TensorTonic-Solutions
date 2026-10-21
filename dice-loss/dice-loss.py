import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    p=np.array(p)
    y=np.array(y)
    intersection=np.sum(p*y)
    dice=(2*intersection+eps)/(np.sum(p)+np.sum(y)+eps)
    return float(1-dice)