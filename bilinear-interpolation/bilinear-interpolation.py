def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    # Write code here
    Bilinear=[]
    h=len(image)
    w=len(image[0])
    
    for i in range(0,new_h):
        row=[]
        for j in range(0,new_w):
            if new_h==1:
                y=0
            else:
                y=i*(h-1)/(new_h-1)

            if new_w==1:
                x=0
            else:
                x=j*(w-1)/(new_w-1)

            y0=int(y)
            y1=min(y0+1,h-1)
            x0=int(x)
            x1=min(x0+1,w-1)

            vO=image[y0][x0]*(1-(x-x0))+image[y0][x1]*(x-x0)
            v1=image[y1][x0]*(1-(x-x0))+image[y1][x1]*(x-x0)
            Oij=vO*(1-(y-y0))+v1*(y-y0)

            row.append(Oij)

        Bilinear.append(row)

    return Bilinear