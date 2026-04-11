import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    im = np.asarray(image)
    ke = np.asarray(kernel)

    H, W = im.shape
    kH, kW = ke.shape

    # padding
    if padding > 0:
        im = np.pad(im, ((padding, padding), (padding, padding)), mode='constant')

    # output size
    H_out = (H + 2*padding - kH)//stride + 1
    W_out = (W + 2*padding - kW)//stride + 1

    output = np.zeros((H_out, W_out))

    # convolution
    for i in range(H_out):
        for j in range(W_out):
            output[i][j] = np.sum(im[i*stride:i*stride+kH, j*stride:j*stride+kW] * ke)

    return output.tolist()