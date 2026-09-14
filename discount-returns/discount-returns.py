def discount_returns(rewards: list, gamma: float) -> list:
    """
    Returns the discounted return at every timestep.
    """
    # Write code here
    Gt=[0]*(len(rewards))
    G=0

    for i in reversed(range(len(rewards))):
        G=rewards[i]+(gamma*G)
        Gt[i]=G

    return Gt