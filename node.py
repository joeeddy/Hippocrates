import random
import math

NODE_ROLES = ['explorer', 'conserver', 'connector', 'innovator']

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = random.random()
        self.memory = [self.state]
        self.energy = random.uniform(0.4, 1.6)
        self.learning_rate = random.uniform(0.01, 0.55)
        self.plasticity = random.uniform(0.01, 0.15)
        self.radius = random.choice([1, 2, 3, 4])
        self.role = random.choice(NODE_ROLES)
        self.rule_type = random.choice(['average', 'sigmoid', 'threshold', 'tanh', 'relu', 'sin'])
        self.signal = random.uniform(-1.0, 1.0)
        self.memory_limit = random.randint(16, 36)
        self.next_state = self.state

    def nonlinear_rule(self, value):
        if self.rule_type == 'sigmoid':
            return 1 / (1 + math.exp(-value))
        elif self.rule_type == 'threshold':
            return 1.0 if value > 0.5 else 0.0
        elif self.rule_type == 'tanh':
            return (math.tanh(value) + 1) / 2
        elif self.rule_type == 'relu':
            return max(0.0, value)
        elif self.rule_type == 'sin':
            return (math.sin(value * math.pi) + 1) / 2
        return value

    def update(self, neighbors, global_mean, global_entropy, global_diversity, local_signals):
        if neighbors:
            neighbor_states = [n.state for n in neighbors]
            avg_state = sum(neighbor_states) / len(neighbor_states)
            self.memory.append(self.state)
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)
            memory_effect = sum(self.memory) / len(self.memory)
            mutation = random.uniform(-0.15, 0.15)
            influence = (
                0.3 * avg_state +
                0.2 * memory_effect +
                0.15 * self.energy +
                0.1 * global_mean +
                0.1 * global_entropy +
                0.15 * global_diversity
            )
            if local_signals:
                influence += 0.1 * (sum(local_signals) / len(local_signals))

            # Meta-learning: adapt learning/plasticity/radius/memory_limit
            if abs(self.state - global_mean) > 0.14 or global_entropy > 2.2:
                self.learning_rate = min(0.65, self.learning_rate + self.plasticity * 0.7)
                if self.energy > 1.3:
                    self.radius = min(4, self.radius + 1)
                self.memory_limit = min(48, self.memory_limit + 1)
                self.plasticity = min(0.2, self.plasticity + 0.01)
            else:
                self.learning_rate = max(0.01, self.learning_rate - self.plasticity * 0.7)
                self.radius = max(1, self.radius - 1)
                self.memory_limit = max(10, self.memory_limit - 1)
                self.plasticity = max(0.01, self.plasticity - 0.005)

            # Role-based and extra dynamic bias
            if self.role == 'explorer':
                influence += mutation * 1.4
            elif self.role == 'conserver':
                influence += (self.state - influence) * 0.25
            elif self.role == 'connector' and local_signals:
                influence += 0.25 * (sum(local_signals) / len(local_signals))
            elif self.role == 'innovator':
                influence += math.sin(sum(neighbor_states)) * 0.25

            # Signal is more dynamic
            self.signal = (math.sin(self.state * 2 * math.pi) +
                           math.cos(self.energy * math.pi) +
                           random.uniform(-0.2, 0.2))
            state_update = (1 - self.learning_rate) * self.state + self.learning_rate * (influence + mutation)
            self.next_state = max(0.0, min(1.0, self.nonlinear_rule(state_update)))
        else:
            self.next_state = self.state

    def commit(self):
        self.state = self.next_state
        self.energy *= random.uniform(0.95, 1.05)
        self.energy = max(0.08, min(3.0, self.energy))

    def display_state(self):
        return f"{self.state:.2f}"
