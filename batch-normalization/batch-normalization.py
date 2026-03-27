# import numpy as np
# from math import pow,sqrt

# def batch_norm_forward(x, gamma, beta, eps=1e-5):
#     """
#     Forward-only BatchNorm for (N,D) or (N,C,H,W).
#     """
#     # Write code here
#     ans=[]
#     if x.shape==2:
#         N,D = x.shape
#         m = N
#         Mean = 0
#         for i in range(m):
#             Mean = Means + x[i]
#         Mean = Mean/m

#         Var = 0
#         for i in range(m):
#             var = var + pow((x[i]-Mean),2)
#         Var = Var/m

#         temp = []
#         for i in range(m):
#             curr = []
#             for j in range(D):
#                 curr[j] = ( x[i] - Mean )/( sqrt(var + eps))
#             temp[i] = curr

#         for i in range(m):
#             for j in range(D):
#                 ans[i][j] = gamma*temp[i][j] + beta
        
#     else:
#         N,C,H,W = x.shape
#         m = N
#         Mean = 0
#         for i in range(m):
#             Mean = Means + x[i]
#         Mean = Mean/m

#         Var = 0
#         for i in range(m):
#             var = var + pow((x[i]-Mean),2)
#         Var = Var/m

#         temp = []
#         for i in range(m):
#             curr = []
#             for j in range(D):
#                 curr[j] = ( x[i] - Mean )/( sqrt(var + eps))
#             temp[i] = curr

#         for i in range(m):
#             for j in range(D):
#                 ans[i][j] = gamma*temp[i][j] + beta

#     return ans
    
#     pass

import numpy as np
from math import pow, sqrt

def batch_norm_forward(x, gamma, beta, eps=1e-5):

    x = np.asarray(x)
    ans = []

    if x.ndim == 2:
        N, D = x.shape

        # mean per feature
        Mean = [0]*D
        for j in range(D):
            for i in range(N):
                Mean[j] += x[i][j]
            Mean[j] /= N

        # variance per feature
        Var = [0]*D
        for j in range(D):
            for i in range(N):
                Var[j] += pow((x[i][j] - Mean[j]), 2)
            Var[j] /= N

        # normalize
        temp = [[0]*D for _ in range(N)]
        for i in range(N):
            for j in range(D):
                temp[i][j] = (x[i][j] - Mean[j]) / (sqrt(Var[j] + eps))

        # scale + shift
        ans = [[0]*D for _ in range(N)]
        for i in range(N):
            for j in range(D):
                ans[i][j] = gamma[j]*temp[i][j] + beta[j]

    else:
        N, C, H, W = x.shape

        # mean per channel
        Mean = [0]*C
        for c in range(C):
            for n in range(N):
                for h in range(H):
                    for w in range(W):
                        Mean[c] += x[n][c][h][w]
            Mean[c] /= (N*H*W)

        # variance per channel
        Var = [0]*C
        for c in range(C):
            for n in range(N):
                for h in range(H):
                    for w in range(W):
                        Var[c] += pow((x[n][c][h][w] - Mean[c]), 2)
            Var[c] /= (N*H*W)

        # normalize
        temp = [[[[0]*W for _ in range(H)] for _ in range(C)] for _ in range(N)]
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        temp[n][c][h][w] = (x[n][c][h][w] - Mean[c]) / (sqrt(Var[c] + eps))

        # scale + shift
        ans = [[[[0]*W for _ in range(H)] for _ in range(C)] for _ in range(N)]
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        ans[n][c][h][w] = gamma[c]*temp[n][c][h][w] + beta[c]

    return np.array(ans)