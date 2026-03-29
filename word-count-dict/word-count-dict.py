def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    dict = {}
    n = len(sentences)
    
    for i in range(n):
        for str in sentences[i]:
            if str in dict:
                dict[str] += 1
            else:
                dict[str] = 1
    
    return dict
    pass