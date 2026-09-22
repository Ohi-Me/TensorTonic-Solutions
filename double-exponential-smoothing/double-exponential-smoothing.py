def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    """
    Returns the smoothed level at every time step.
    """

    if not series:
        return []
    # Initialize level with the first observation
    level=series[0]
    # Initialize trend using the difference between first two values
    trend=series[1] - series[0]
    # Store the smoothed levels
    smoothed=[level]

    for i in range(1, len(series)):

        # Save previous level before updating it
        previous_level = level
        # Update level
        level = alpha * series[i] + (1 - alpha) * (level + trend)
        # Update trend
        trend = beta * (level - previous_level) + (1 - beta) * trend
        # Store current smoothed level
        smoothed.append(level)

    return smoothed