# node.py
import random

class Node:
    def __init__(self, x, y):
        self.x = x  # Position in grid
        self.y = y
        self.state = random.random()  # Initial state: random float between 0 and 1
        self.next_state = self.state  # Buffer for next state

    def update(self, neighbors):
        # Learning rule: average neighbors' states + small random mutation
        if neighbors:
            avg_state = sum(n.state for n in neighbors) / len(neighbors)
            mutation = random.uniform(-0.05, 0.05)  # Small random change
            self.next_state = max(0.0, min(1.0, avg_state + mutation))
        else:
            self.next_state = self.state  # No neighbors, keep state

    def commit(self):
        # Apply the computed next state
        self.state = self.next_state
