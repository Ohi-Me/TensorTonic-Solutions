import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    # distInd=[[]]
    # n=len(X_train)
    # m=len(X_test)
    # for i in range(n):
    #     dist=0
    #     for j in range(m):
    #         int Currdist=pow((X_test[j]-X_train[i]),2);
    #         dist=dist+Currdist;
    #     distInd[dist]=i

    # sorted(distInd)
    # i=0
    # while(i<k):
    #     ans[i]=distInd[i].second
    #     i=i+1

    # return ans

    if len(X_test)==0:
        return np.empty((0,k),dtype=int)
    distInd=[]
    n=len(X_train)
    m=len(X_test)
    for i in range(m):
        temp=[]
        for j in range(n):
            dist=0
            xt=np.atleast_1d(X_test[i])
            yt=np.atleast_1d(X_train[j])
            for d in range(len(xt)):
                dist+=(xt[d]-yt[d])**2
            temp.append((dist,j))
        temp.sort()
        ans=[]
        x=0
        while x<min(k,n):
            ans.append(temp[x][1])
            x+=1
        while x<k:
            ans.append(-1)
            x+=1
        distInd.append(ans)
    return np.array(distInd,dtype=int)
    pass