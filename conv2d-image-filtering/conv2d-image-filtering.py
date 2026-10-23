def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    h,w=len(image),len(image[0])
    kh,kw=len(kernel),len(kernel[0])
    oh=(h+2*padding-kh)//stride+1
    ow=(w+2*padding-kw)//stride+1
    res=[[0]*ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            for a in range(kh):
                for b in range(kw):
                    x=i*stride+a-padding
                    y=j*stride+b-padding
                    if 0<=x<h and 0<=y<w:
                        res[i][j]+=image[x][y]*kernel[a][b]
    return res