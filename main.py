# main.py
from network import Network

def main():
    # Initialize a 10x10 grid of nodes
    network = Network(grid_size=10)
    
    # Run simulation for 100 iterations
    for step in range(100):
        print(f"\nStep {step}:")
        network.update()  # Update all nodes
        network.print_states()  # Display grid of states

if __name__ == "__main__":
    main()
