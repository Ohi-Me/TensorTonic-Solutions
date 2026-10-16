def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    matrix=[[] for _ in range(k)]
    for i in range(k):
        for j in range(len(points)):
            if assignments[j]==i:
                matrix[i].append(points[j])

    for i in range(k):
        if len(matrix[i])>0:
            centroid=[]
            for j in range(len(matrix[i][0])):
                total=0
                for l in range(len(matrix[i])):
                    total+=matrix[i][l][j]

                centroid.append(total/len(matrix[i]))

            matrix[i]=centroid
        else:
            matrix[i]=[0.0 for _ in range(len(points[0]))]

    return matrix