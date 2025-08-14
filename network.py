from node import Node
import numpy as np

class Network:
    def __init__(self, grid_size):
        self.size = grid_size
        self.grid = [[Node(i, j) for j in range(grid_size)] for i in range(grid_size)]

    def get_neighbors(self, x, y, radius=1):
        neighbors = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    neighbors.append(self.grid[nx][ny])
        return neighbors

    def global_state_mean(self):
        flat = [self.grid[i][j].state for i in range(self.size) for j in range(self.size)]
        return sum(flat) / len(flat) if flat else 0.0

    def global_entropy(self):
        flat = [self.grid[i][j].state for i in range(self.size) for j in range(self.size)]
        hist, _ = np.histogram(flat, bins=10, range=(0, 1), density=True)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def global_diversity(self):
        flat = [round(self.grid[i][j].state, 1) for i in range(self.size) for j in range(self.size)]
        unique = set(flat)
        return len(unique) / len(flat) if flat else 0

    def update(self):
        global_mean = self.global_state_mean()
        global_entropy = self.global_entropy()
        global_diversity = self.global_diversity()
        for i in range(self.size):
            for j in range(self.size):
                node = self.grid[i][j]
                neighbors = self.get_neighbors(i, j, radius=node.radius)
                local_signals = [n.signal for n in neighbors]
                node.update(neighbors, global_mean, global_entropy, global_diversity, local_signals)
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].commit()
                
    def print_states(self):
        for i in range(self.size):
            row = [self.grid[i][j].display_state() for j in range(self.size)]
            print(" ".join(row))
