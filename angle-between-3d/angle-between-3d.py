import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    v=np.array(v)
    w=np.array(w)
    
    norm_v=np.linalg.norm(v)
    norm_w=np.linalg.norm(w)
    
    if norm_v==0 or norm_w==0:
        return float('nan')
        
    cosine=np.dot(v,w)/(norm_v*norm_w)
    
    if cosine>1:
        cosine=1
    if cosine<-1:
        cosine=-1
        
    angle=np.arccos(cosine)
    return float(angle)