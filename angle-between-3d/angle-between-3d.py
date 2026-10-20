import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    v=np.array(v,dtype=float)
    w=np.array(w,dtype=float)
    
    norm_v=np.linalg.norm(v)
    norm_w=np.linalg.norm(w)
    
    if norm_v==0 or norm_w==0:
        return float(np.nan)
        
    cosine=np.dot(v,w)/(norm_v*norm_w)
    cosine=np.clip(cosine,-1.0,1.0)
    
    angle=np.arccos(cosine)
    return float(angle)