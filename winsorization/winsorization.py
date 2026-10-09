def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    """
    Returns values clipped to the interpolated percentile bounds.
    """
    # Write code here
    k=[]
    a=sorted(values)
    n=len(a)

    k=(n-1)*lower_pct/100
    i=int(k)
    lower=a[i]+(k-i)*(a[min(i+1,n-1)]-a[i])

    k=(n-1)*upper_pct/100
    i=int(k)
    upper=a[i]+(k-i)*(a[min(i+1,n-1)]-a[i])

    ans=[]
    for i in range(0,len(values)):
        if values[i]<lower:
            ans.append(lower)
        elif values[i]>upper:
            ans.append(upper)
        else:
            ans.append(values[i])

    return ans