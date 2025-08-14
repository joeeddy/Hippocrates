import numpy as np

def box_count(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return len(np.where(S > 0)[0])

def fractal_dimension(Z):
    """
    Estimate the fractal dimension of a 2D binary array Z using box-counting.
    Z: 2D numpy array with values 0 or 1
    Returns: fractal dimension (float)
    """
    assert(len(Z.shape) == 2)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    n = int(n)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))
    coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
    return -coeffs[0]
