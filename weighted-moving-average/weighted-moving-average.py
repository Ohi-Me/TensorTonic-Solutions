import numpy as np

def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    # Write code here
    values=np.asarray(values)
    weights=np.asarray(weights)
    wma=[]

    for i in range(0,len(values)-len(weights)+1):
        total=0
        for j in range(len(weights)):
            total+=(values[i+j]*weights[j])
        total/=(sum(weights))
        wma.append(total)
    return wma