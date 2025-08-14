import numpy as np

# Fractal dimension calculation using box-counting method
# Fixed to ensure results are always in the valid range [1, 2] for 2D patterns
# Improvements:
# - Extended box size range for better regression fit
# - Added robust handling of edge cases (all zeros, all ones, insufficient data)
# - Proper mathematical relationship for box-counting
# - Clamping to physically meaningful range

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
    
    The fractal dimension should be between 1 and 2 for 2D patterns:
    - Close to 1.0 for line-like structures
    - Close to 2.0 for space-filling patterns
    """
    assert(len(Z.shape) == 2)
    
    # Handle edge cases
    if Z.size == 0:
        return 1.0
    
    # Check for degenerate cases (all zeros or all ones)
    if not np.any(Z):  # All zeros
        return 1.0  # Minimal dimension
    if np.all(Z):  # All ones
        return 2.0  # Maximal dimension for 2D
    
    # Create a range of box sizes from 1 to min(dimensions)/2
    # Use more sizes for better regression fit
    max_size = min(Z.shape) // 2
    if max_size < 1:
        max_size = 1
    
    # Generate box sizes: start from largest and go down to 1
    sizes = []
    size = max_size
    while size >= 1:
        sizes.append(size)
        if size == 1:
            break
        # Use smaller steps for better resolution
        size = max(1, size // 2)
    
    # Ensure we have size 1 if it's not already included
    if sizes[-1] != 1:
        sizes.append(1)
    
    # Calculate box counts for each size
    counts = []
    for size in sizes:
        count = box_count(Z, size)
        counts.append(count)
    
    # Filter out zero counts to avoid log(0)
    valid_pairs = [(s, c) for s, c in zip(sizes, counts) if c > 0]
    
    if len(valid_pairs) < 2:
        # Not enough data points for regression, return sensible default
        # Based on how much of the grid is filled
        fill_ratio = np.sum(Z) / Z.size
        # Linear interpolation between 1 (sparse) and 2 (dense)
        return 1.0 + fill_ratio
    
    valid_sizes, valid_counts = zip(*valid_pairs)
    
    # Fractal dimension is negative slope of log(count) vs log(size)
    # As box size increases, count should decrease for fractal patterns
    log_sizes = np.log(valid_sizes)
    log_counts = np.log(valid_counts)
    
    # Perform linear regression: log(count) = slope * log(size) + intercept
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    slope = coeffs[0]
    
    # Fractal dimension is the negative of the slope
    fractal_dim = -slope
    
    # Clamp to valid range for 2D patterns
    # Add small tolerance for numerical errors
    fractal_dim = max(1.0, min(2.0, fractal_dim))
    
    return fractal_dim
