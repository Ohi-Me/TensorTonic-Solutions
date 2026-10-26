def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    # Write code here
    ema=[values[0]] # Initialize with first value
    for i in range(1,len(values)):
        ema.append(alpha*values[i]+(1-alpha)*ema[i-1])
    return ema
