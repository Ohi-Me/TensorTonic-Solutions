import math

def evaluate_shadow(production_log: list, shadow_log: list, criteria: dict) -> dict:
    """
    Returns a dictionary with the promotion decision and metrics.
    """
    # Write code here
    correctPro=0;
    for i in range(len(production_log)):
        if production_log[i]["prediction"]==production_log[i]["actual"]:
            correctPro+=1

    correctShadow=0;
    for i in range(len(shadow_log)):
        if shadow_log[i]["prediction"]==shadow_log[i]["actual"]:
            correctShadow+=1

    accPro=correctPro/len(production_log)
    accShadow=correctShadow/len(shadow_log)

    accGain=accShadow-accPro

    latencies=[]
    for i in range(len(shadow_log)):
        latencies.append(shadow_log[i]["latency_ms"])

    latencies.sort()

    index=math.ceil(0.95*len(latencies))-1
    shadowP95=latencies[index]

    agreement=0
    for i in range(len(production_log)):
        if production_log[i]["prediction"]==shadow_log[i]["prediction"]:
            agreement+=1

    agreementRate=agreement/len(production_log)

    promote=(
        accGain>=criteria["min_accuracy_gain"]
        and shadowP95<=criteria["max_latency_p95"]
        and agreementRate>=criteria["min_agreement_rate"]
    )

    return {
        "promote":promote,
        "metrics":{
            "shadow_accuracy":accShadow,
            "production_accuracy":accPro,
            "accuracy_gain":accGain,
            "shadow_latency_p95":shadowP95,
            "agreement_rate":agreementRate
        }
    }
