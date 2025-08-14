#!/usr/bin/env python3

import numpy as np
from fractal import box_count

def test_box_count():
    """Test the original box_count function."""
    
    # Simple 4x4 test pattern
    pattern = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1], 
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    
    print("Test pattern:")
    print(pattern)
    
    for k in [1, 2, 4]:
        count = box_count(pattern, k)
        print(f"box_count with k={k}: {count}")
        
        # Let's debug what add.reduceat does
        print(f"  Shape: {pattern.shape}")
        
        # Manual calculation for verification
        h, w = pattern.shape
        manual_count = 0
        print(f"  Manual boxes for k={k}:")
        for i in range(0, h, k):
            for j in range(0, w, k):
                box = pattern[i:i+k, j:j+k]
                has_content = np.any(box)
                if has_content:
                    manual_count += 1
                print(f"    box[{i}:{i+k}, {j}:{j+k}] = {box.sum()} (any: {has_content})")
        
        print(f"  Manual count: {manual_count}")
        print()

if __name__ == "__main__":
    test_box_count()