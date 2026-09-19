def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    # Write code here
    currDay=0
    currBudget=config["budget"]
    currRetrain=0

    trigerDrift=False
    trigerPerformance=False
    trigerStateness=False

    ans=[]
    
    for i in range(0,len(daily_stats)):
        if daily_stats[i]["drift_score"] > config["drift_threshold"]:
            trigerDrift=True
        if daily_stats[i]["performance"] < config["performance_threshold"]:
            trigerPerformance=True

        currDay+=1

        if currDay % config["max_staleness"] == 0:
            trigerStateness=True

        if (trigerDrift or trigerPerformance or trigerStateness) and (currDay-currRetrain >= config["cooldown"] or currRetrain == 0) and currBudget >= config["retrain_cost"]:
            ans.append((i+1));
            currBudget-=config["retrain_cost"]
            currRetrain=currDay
            trigerDrift=False
            trigerPerformance=False
            trigerStateness=False

    return ans