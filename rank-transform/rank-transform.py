def rank_transform(values: list) -> list:
    sorted_values=sorted(values)
    N=len(values)
    ranks={}
    i=0

    while i<N:
        j=i

        while j<N and sorted_values[j]==sorted_values[i]:
            j+=1

        rank=((i+1)+j)/2.0
        ranks[sorted_values[i]]=rank

        i=j

    return [ranks[value] for value in values]