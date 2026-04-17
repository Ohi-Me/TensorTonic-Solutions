import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    y_true = list(y_true)
    y_pred = list(y_pred)

    n = len(y_true)

    if num_classes is None:
        K = max(max(y_true), max(y_pred)) + 1
    else:
        K = num_classes

    conf = [[0]*K for _ in range(K)]

    for i in range(n):
        conf[y_true[i]][y_pred[i]] += 1

    if normalize == 'true':
        for i in range(K):
            s = 0
            for j in range(K):
                s += conf[i][j]
            for j in range(K):
                conf[i][j] = conf[i][j] / (s + 1e-12)

    elif normalize == 'pred':
        for j in range(K):
            s = 0
            for i in range(K):
                s += conf[i][j]
            for i in range(K):
                conf[i][j] = conf[i][j] / (s + 1e-12)

    elif normalize == 'all':
        s = 0
        for i in range(K):
            for j in range(K):
                s += conf[i][j]
        for i in range(K):
            for j in range(K):
                conf[i][j] = conf[i][j] / (s + 1e-12)

    return conf