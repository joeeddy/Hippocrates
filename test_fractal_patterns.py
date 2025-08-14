#!/usr/bin/env python3

import numpy as np

def test_fractal_patterns():
    """Test fractal dimension calculation on known patterns."""
    
    # Test pattern 1: Checkerboard (should be close to 2.0 for a filled pattern)
    checkerboard = np.zeros((8, 8))
    checkerboard[::2, ::2] = 1
    checkerboard[1::2, 1::2] = 1
    
    # Test pattern 2: Line (should be close to 1.0)
    line = np.zeros((8, 8))
    line[4, :] = 1  # horizontal line
    
    # Test pattern 3: Single point
    point = np.zeros((8, 8))
    point[4, 4] = 1
    
    # Test pattern 4: Full grid
    full = np.ones((8, 8))
    
    patterns = [
        ("Checkerboard", checkerboard),
        ("Line", line),
        ("Point", point),
        ("Full", full)
    ]
    
    for name, pattern in patterns:
        print(f"\n=== {name} ===")
        print(pattern.astype(int))
        
        # Calculate fractal dimension with improved algorithm
        fd = improved_fractal_dimension(pattern.astype(int))
        print(f"Fractal dimension: {fd:.3f}")

def improved_fractal_dimension(Z):
    """
    Improved fractal dimension calculation with better size range.
    """
    assert(len(Z.shape) == 2)
    
    # Use all possible sizes from 1 to half the minimum dimension
    max_size = min(Z.shape) // 2
    if max_size < 1:
        max_size = 1
    
    sizes = list(range(1, max_size + 1))
    sizes.reverse()  # Start from largest to smallest
    
    counts = []
    for size in sizes:
        count = box_count_improved(Z, size)
        counts.append(count)
        print(f"  size={size}, count={count}")
    
    if len(counts) < 2:
        print("  Warning: Not enough data points for regression")
        return 2.0  # Default to maximum dimension for 2D
    
    # Remove zero counts to avoid log(0)
    valid_indices = [i for i, c in enumerate(counts) if c > 0]
    if len(valid_indices) < 2:
        print("  Warning: Not enough non-zero counts")
        return 2.0
    
    valid_sizes = [sizes[i] for i in valid_indices]
    valid_counts = [counts[i] for i in valid_indices]
    
    # The fractal dimension is the negative slope of log(count) vs log(1/size)
    # But we want log(count) vs log(size), so the slope is negative fractal dimension
    log_sizes = np.log(valid_sizes)
    log_counts = np.log(valid_counts)
    
    print(f"  log(sizes): {log_sizes}")
    print(f"  log(counts): {log_counts}")
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    # The slope is negative of fractal dimension (as size increases, count decreases)
    fractal_dim = -coeffs[0]
    
    print(f"  slope: {coeffs[0]}, fractal_dim: {fractal_dim}")
    
    # Clamp to reasonable range for 2D
    return max(1.0, min(2.0, fractal_dim))

def box_count_improved(Z, k):
    """
    Improved box counting that handles edge cases better.
    """
    if k >= min(Z.shape):
        return 1 if np.any(Z) else 0
    
    # Create boxes and count non-empty ones
    h, w = Z.shape
    count = 0
    
    for i in range(0, h, k):
        for j in range(0, w, k):
            box = Z[i:i+k, j:j+k]
            if np.any(box):
                count += 1
    
    return count

if __name__ == "__main__":
    test_fractal_patterns()