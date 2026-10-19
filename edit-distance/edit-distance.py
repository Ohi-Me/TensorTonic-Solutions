def edit_distance(s1: str, s2: str) -> int:
    m=len(s1)
    n=len(s2)
    prev=list(range(n+1))

    for i in range(1,m+1):
        curr=[i]+[0]*n

        for j in range(1,n+1):
            if s1[i-1]==s2[j-1]:
                curr[j]=prev[j-1]
            else:
                curr[j]=1+min(prev[j],curr[j-1],prev[j-1])

        prev=curr

    return prev[n]