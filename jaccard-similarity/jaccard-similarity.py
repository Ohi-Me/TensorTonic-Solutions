def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    a = list(set(set_a))
    b = set(set(set_b))
    both=0
    n=len(a)
    m=len(b)
    for i in range(n):
        if a[i] in b:
            both+=1
    notcommon=n+m-both
    # return both/notcommon
    return both / notcommon if notcommon != 0 else 0