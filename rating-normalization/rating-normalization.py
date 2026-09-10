def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    # Write code here
    ans=[]
    for i in range(0,len(matrix)):
        sum=0
        cnt=0
        for j in range(0,len(matrix[0])):
            if matrix[i][j]==0:
                continue
            cnt+=1
            sum+=matrix[i][j]

        if sum==0:
            ans.append([0]*len(matrix[0]))
            continue
            
        mean=sum/cnt
        curr=[]
        for j in range(0,len(matrix[0])):
            if matrix[i][j]==0:
                curr.append(0)
            else:
                curr.append((matrix[i][j]-mean))
        ans.append(curr)
    return ans
    pass