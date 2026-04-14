import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    image=np.asarray(image)
    h,w=image.shape

    bin=[0]*256
    for i in range(h):
        for j in range(w):
            curr=image[i][j]
            bin[curr]+=1
    return bin