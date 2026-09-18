import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """

    X = np.array(X, dtype=float)
    Pij = []

    # Loop through columns
    for i in range(X.shape[1]):
        row = []

        for j in range(X.shape[1]):

            x = X[:, i]
            y = X[:, j]

            # Mean
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Numerator
            numerator = np.sum((x - x_mean) * (y - y_mean))

            # Denominator
            denominator = np.sqrt(
                np.sum((x - x_mean) ** 2) *
                np.sum((y - y_mean) ** 2)
            )

            # If denominator is 0, correlation is NaN
            if denominator == 0:
                corr = np.nan
            else:
                corr = numerator / denominator

            row.append(corr)

        Pij.append(row)

    return np.array(Pij)