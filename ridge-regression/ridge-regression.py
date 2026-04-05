import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    #transpose
    X=np.asarray(X)
    y=np.asarray(y)
    n=len(X);
    m=len(X[0])
    Xt=np.zeros((m,n))
    for i in range(n):
        t=[0]*m
        for j in range(m):
            t[j]=X[i][j]
        for j in range(m):
            Xt[j][i]=t[j]

    #multiplication
    z=np.zeros((m,m))
    for i in range(m):
        t=[0]*n
        for j in range(n):
            t[j]=Xt[i][j]
        for j in range(m):
            sum=0
            for k in range(n):
                sum+=t[k]*X[k][j];
            z[i][j]=sum

    z1=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if(i==j):
                z1[i][j]=lam
            else:
                z1[i][j]=0

    #addition
    for i in range(m):
        for j in range(m):
            z[i][j]+=z1[i][j]

    #inverse
    Z_inverse=np.linalg.inv(np.array(z))

    #final
    z2=[0]*m
    for i in range(m):
        t=[0]*n
        for j in range(n):
            t[j]=Xt[i][j]
        sum=0
        for j in range(n):
            sum+=t[j]*y[j];
        z2[i]=sum

    #final
    ans=[0]*m
    for i in range(m):
        t=[0]*m
        for j in range(m):
            t[j]=Z_inverse[i][j]
        sum=0
        for j in range(m):
            sum+=t[j]*z2[j];
        ans[i]=sum

    return ans