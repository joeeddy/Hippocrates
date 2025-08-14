#!/usr/bin/env python3

import numpy as np
from network import Network
from fractal import fractal_dimension, box_count

def debug_fractal_calculation():
    """Debug the fractal dimension calculation to understand the issue."""
    
    # Create a network and run a few steps
    network = Network(grid_size=10)
    network.update()
    
    # Get the states and create binary grid
    states = np.array([[node.state for node in row] for row in network.grid])
    binary = (states > 0.5).astype(int)
    
    print("=== DEBUG FRACTAL DIMENSION CALCULATION ===")
    print(f"Original states shape: {states.shape}")
    print(f"States min/max: {states.min():.3f} / {states.max():.3f}")
    print(f"States mean: {states.mean():.3f}")
    
    print(f"\nBinary grid (threshold=0.5):")
    print(binary)
    print(f"Binary grid shape: {binary.shape}")
    print(f"Binary sum (number of 1s): {binary.sum()}")
    print(f"Binary zeros: {(binary == 0).sum()}")
    print(f"Binary ones: {(binary == 1).sum()}")
    
    # Debug the fractal calculation step by step
    print(f"\n=== FRACTAL CALCULATION STEPS ===")
    
    # Reproduce the fractal_dimension calculation with debug output
    Z = binary
    assert(len(Z.shape) == 2)
    
    p = min(Z.shape)
    print(f"p (min dimension): {p}")
    
    n = 2**np.floor(np.log2(p))
    n = int(n)
    print(f"n (largest power of 2 <= p): {n}")
    
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    print(f"sizes array: {sizes}")
    
    counts = []
    for size in sizes:
        count = box_count(Z, size)
        counts.append(count)
        print(f"size={size}, count={count}")
    
    print(f"counts array: {counts}")
    
    if len(counts) < 2:
        print("ERROR: Not enough data points for linear regression!")
        return
    
    # Check the log values
    log_inv_sizes = np.log(1/sizes)
    log_counts = np.log(counts)
    
    print(f"1/sizes: {1/sizes}")
    print(f"log(1/sizes): {log_inv_sizes}")
    print(f"log(counts): {log_counts}")
    
    # Check for invalid log values
    if np.any(np.isnan(log_counts)) or np.any(np.isinf(log_counts)):
        print("ERROR: Invalid log(counts) values detected!")
        print(f"NaN in log_counts: {np.any(np.isnan(log_counts))}")
        print(f"Inf in log_counts: {np.any(np.isinf(log_counts))}")
    
    coeffs = np.polyfit(log_inv_sizes, log_counts, 1)
    print(f"polyfit coefficients: {coeffs}")
    print(f"slope (negative fractal dimension): {coeffs[0]}")
    print(f"fractal dimension: {-coeffs[0]}")
    
    # Test with different thresholds
    print(f"\n=== TESTING DIFFERENT THRESHOLDS ===")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary_test = (states > threshold).astype(int)
        fd_test = fractal_dimension(binary_test)
        ones_count = binary_test.sum()
        print(f"threshold={threshold}: ones={ones_count}, fractal_dim={fd_test:.3f}")

if __name__ == "__main__":
    debug_fractal_calculation()