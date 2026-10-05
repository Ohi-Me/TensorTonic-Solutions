def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    best = models[0]

    for model in models[1:]:
        if model["accuracy"] > best["accuracy"]:
            best = model

        elif model["accuracy"] == best["accuracy"]:
            if model["latency"] < best["latency"]:
                best = model

            elif model["latency"] == best["latency"]:
                if model["timestamp"] > best["timestamp"]:
                    best = model

    return best["name"]