from network import Network
from fractal import fractal_dimension
import numpy as np

def main():
    network = Network(grid_size=10)
    
    for step in range(100):
        print(f"\nStep {step}:")
        network.update()
        network.print_states()
        
        # Fractal dimension
        states = np.array([[node.state for node in row] for row in network.grid])
        binary = (states > 0.5).astype(int)
        fd = fractal_dimension(binary)
        print(f"Fractal dimension: {fd:.2f}")
        
        # Global entropy and other global stats
        entropy = network.global_entropy()
        diversity = network.global_diversity()
        print(f"Global entropy: {entropy:.2f}")
        print(f"Global diversity: {diversity:.2f}")

if __name__ == "__main__":
    main()
