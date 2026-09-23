def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    x_Mean=sum(series)/len(series)
    total_Variance=0
    
    for x in series:
        total_Variance+=pow((x-x_Mean),2)

    lag_K=[]
    
    if total_Variance==0:
        lag_K.append(1.0)
        
        for k in range(1,max_lag+1):
            lag_K.append(0.0)
        
        return lag_K
        
    for k in range(max_lag+1):
        lag_First=0

        for i in range(k,len(series)):
            lag_First+=((series[i]-x_Mean)*(series[i-k]-x_Mean))

        lag_K.append(lag_First/total_Variance)

    return lag_K