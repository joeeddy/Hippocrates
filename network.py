# network.py
from node import Node

class Network:
    def __init__(self, grid_size):
        self.size = grid_size
        # Create a 2D grid of nodes
        self.grid = [[Node(i, j) for j in range(grid_size)] for i in range(grid_size)]

    def get_neighbors(self, x, y):
        # Return list of neighboring nodes (up, down, left, right)
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-directional grid
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append(self.grid[nx][ny])
        return neighbors

    def update(self):
        # Update all nodes based on neighbors
        for i in range(self.size):
            for j in range(self.size):
                neighbors = self.get_neighbors(i, j)
                self.grid[i][j].update(neighbors)
        # Commit updates (synchronous)
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].commit()

    def print_states(self):
        # Print grid of states for visualization
        for i in range(self.size):
            row = [f"{self.grid[i][j].state:.2f}" for j in range(self.size)]
            print(" ".join(row))
