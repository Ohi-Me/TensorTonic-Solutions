
import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """Returns a generator of (X_batch, y_batch) tuples."""
    X, y = np.asarray(X), np.asarray(y)
    indices = np.random.default_rng(seed).permutation(len(X))

    for i in range(0, len(X), batch_size):
        idx = indices[i:i+batch_size]
        if drop_last and len(idx) < batch_size:
            break
        yield X[idx], y[idx]
    pass
