
import math

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    if not candidate:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        cand = {}
        ref = {}

        for i in range(len(candidate) - n + 1):
            t = tuple(candidate[i:i+n])
            if t in cand:
                cand[t] += 1
            else:
                cand[t] = 1

        for i in range(len(reference) - n + 1):
            t = tuple(reference[i:i+n])
            if t in ref:
                ref[t] += 1
            else:
                ref[t] = 1

        total = sum(cand.values())
        match = 0

        for t in cand:
            if t in ref:
                match += min(cand[t], ref[t])

        if total == 0 or match == 0:
            return 0.0

        precisions.append(match / total)

    c = len(candidate)
    r = len(reference)

    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    score = math.exp(sum(math.log(p) for p in precisions) / max_n)

    return bp * score
