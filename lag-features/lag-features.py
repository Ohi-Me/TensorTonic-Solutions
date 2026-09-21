def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    # Write code here
    lag=[]
    whichLag=max(lags)

    for i in range(whichLag, len(series)):
        idx=i
        curr=[]

        for j in lags:
            curr.append(series[idx-j])

        lag.append(curr)

    return lag