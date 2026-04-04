from math import log,exp

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    n = len(prob_distributions)
    H = 0
    for i in range(n):
        H += log(prob_distributions[i][actual_tokens[i]])
    H = -(1/n)*H
    return exp(H)