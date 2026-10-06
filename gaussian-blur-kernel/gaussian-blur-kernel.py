import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here
    center=size//2
    kernel=[]

    for i in range(size):
        row=[]
        for j in range(size):
            x=i-center
            y=j-center

            value=math.exp(-(x**2 + y**2)/(2*sigma**2))

            row.append(value)
        kernel.append(row)

    total=sum(sum(row) for row in kernel)
    for i in range(size):
        for j in range(size):
            kernel[i][j]/=total

    return kernel
        