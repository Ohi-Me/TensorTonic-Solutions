def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    dict={}
    for sentence in sentences:
        for word in sentence:
            dict[word]=dict.get(word,0)+1
    return dict
    pass