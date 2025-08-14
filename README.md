# 🧠 Hippocrates

**A Self-Organizing, Self-Learning AI System with Fractal-Emergent Intelligence**

*Inspired by decentralized principles and emergent complexity*

---

## ⚡ Quick Start

### Run Instantly Online
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joeeddy/Hippocrates/)
[![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://joeeddy.github.io/Hippocrates/)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/joeeddy/Hippocrates.git
cd Hippocrates

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py
```

**Requirements:** Python 3.7+ with NumPy

---

## ✨ What is Hippocrates?

Hippocrates is a **self-organizing AI system** that demonstrates emergent intelligence through the interactions of simple autonomous agents (nodes). Each node learns, adapts, and evolves its behavior based on local interactions, creating complex patterns and collective intelligence without central control.

**Key Characteristics:**
- 🧠 **Self-Learning**: Nodes adapt their learning rates, memory, and behavior based on performance
- 🌐 **Self-Organizing**: Complex patterns emerge from simple local interactions
- 📐 **Fractal-Emergent**: System complexity measurable through fractal dimension analysis
- 🔄 **Decentralized**: No central authority—intelligence emerges from the network itself

---

## 🏗️ Architecture Overview

### Core Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **main.py** | Simulation orchestrator | Runs simulation loop, calculates global metrics |
| **network.py** | Grid management | Manages 10×10 node grid, handles neighbor interactions |
| **node.py** | Individual agents | Implements learning, memory, roles, and adaptation |
| **fractal.py** | Pattern analysis | Box-counting fractal dimension calculation |

### System Design

```
    Hippocrates AI System
    ┌─────────────────────────────┐
    │        main.py              │
    │   (Simulation Controller)   │
    └─────────┬───────────────────┘
              │
    ┌─────────▼───────────────────┐
    │      network.py             │
    │   (10×10 Grid Manager)      │
    │  - Global state tracking    │
    │  - Neighbor management      │
    │  - Entropy calculation      │
    └─────────┬───────────────────┘
              │
    ┌─────────▼───────────────────┐
    │       node.py               │
    │   (Individual Agents)       │
    │  - Learning & adaptation    │
    │  - Memory & energy          │
    │  - Role-based behavior      │
    │  - Signal propagation       │
    └─────────┬───────────────────┘
              │
    ┌─────────▼───────────────────┐
    │     fractal.py              │
    │   (Pattern Analysis)        │
    │  - Box-counting algorithm   │
    │  - Complexity measurement   │
    └─────────────────────────────┘
```

---

## 🔬 How It Works

### Node Intelligence
Each node in the 10×10 grid is an **autonomous agent** with:

- **State**: Continuous value (0-1) representing its current condition
- **Memory**: Historical states for pattern recognition
- **Energy**: Dynamic resource affecting interaction range
- **Learning Rate**: Adaptive parameter for behavioral change
- **Plasticity**: Ability to modify learning parameters
- **Role**: Specialized behavior (Explorer, Conserver, Connector, Innovator)
- **Signal**: Dynamic communication with neighbors

### Learning Mechanisms

**1. Meta-Learning Adaptation**
```python
# Nodes adapt their learning capabilities based on performance
if abs(self.state - global_mean) > threshold:
    self.learning_rate += self.plasticity * factor
    self.memory_limit += 1
    self.radius = min(4, self.radius + 1)
```

**2. Role-Based Behavior**
- **Explorers**: Amplify random mutations for discovery
- **Conservers**: Maintain stable states and resist change
- **Connectors**: Bridge different network regions
- **Innovators**: Generate novel patterns through mathematical functions

**3. Multi-Scale Influence**
Nodes integrate information from:
- Immediate neighbors (local patterns)
- Personal memory (temporal patterns)
- Global network state (collective intelligence)
- Energy levels (resource availability)

### Emergent Properties

**Self-Organization**: Complex patterns emerge without central planning
**Collective Intelligence**: Network-wide behaviors exceed individual capabilities  
**Adaptive Complexity**: System complexity measured via fractal dimension
**Dynamic Equilibrium**: Balance between stability and change

---

## 📊 Example Output

```
=== Hippocrates AI Simulation ===
Initial state (Step 0):
0.34 0.75 0.19 0.53 0.87 0.45 0.41 0.33 0.08 0.29
0.21 0.09 0.26 0.17 0.47 0.96 0.80 1.00 0.18 0.48
...

Step 1:
0.00 0.82 0.34 1.00 0.70 0.53 0.99 0.73 0.21 0.57
0.43 0.92 0.52 0.58 0.64 0.71 0.80 0.99 0.00 0.97
...

Fractal dimension: -1.17
Global entropy: 2.81
Global diversity: 0.11

• Each node learns from neighbors and adapts its behavior
• Fractal dimension measures pattern complexity in the network
• Global entropy tracks information distribution  
• Global diversity measures state variety across nodes
```

### Key Metrics Explained

- **Fractal Dimension**: Measures geometric complexity of active patterns
- **Global Entropy**: Information content and unpredictability
- **Global Diversity**: Variety of states across the network

---

## 🧪 Experimentation Guide

### Modify Simulation Parameters

**Grid Size** (in main.py):
```python
network = Network(grid_size=15)  # Try 5, 10, 15, 20
```

**Simulation Length**:
```python
for step in range(200):  # Experiment with longer runs
```

**Node Behavior** (in node.py):
```python
# Adjust learning dynamics
self.learning_rate = random.uniform(0.01, 0.8)  # More/less adaptive
self.plasticity = random.uniform(0.005, 0.3)    # Flexibility range
```

### Research Questions to Explore

1. **Convergence Patterns**: How does network size affect stabilization?
2. **Role Distribution**: What happens with different role proportions?
3. **Memory Effects**: How does memory length influence learning?
4. **Energy Dynamics**: Impact of different energy distributions?
5. **Fractal Evolution**: How does complexity change over time?

---

## 🚀 Getting Started for Developers

### Project Structure
```
Hippocrates/
├── main.py           # Entry point and simulation loop
├── network.py        # Grid and global state management  
├── node.py          # Individual agent implementation
├── fractal.py       # Fractal dimension analysis
├── requirements.txt # Dependencies
└── README.md        # This file
```

### Adding New Features

**Custom Node Roles**:
```python
# Add to NODE_ROLES in node.py
NODE_ROLES = ['explorer', 'conserver', 'connector', 'innovator', 'your_role']
```

**New Metrics**:
```python
# Add to network.py
def your_metric(self):
    # Implement custom analysis
    return computed_value
```

**Different Update Rules**:
```python
# Modify node.py nonlinear_rule() method
def nonlinear_rule(self, value):
    if self.rule_type == 'your_rule':
        return your_transformation(value)
```

---

## 🌍 Philosophy & Vision

Hippocrates embodies the principle that **intelligence emerges from decentralized collaboration**. Inspired by natural systems and distributed networks, it explores how simple interactions can generate complex, adaptive behavior.

**Core Beliefs:**
- Intelligence should be transparent and explainable
- Learning systems should adapt and evolve continuously  
- Decentralized approaches can solve complex problems
- Open science accelerates understanding

**Future Directions:**
- Educational AI that learns with students
- Distributed problem-solving networks
- Transparent AI decision-making systems
- Community-driven AI development

---

## 🤝 Contributing

We welcome contributions! Whether you're interested in:
- 🔬 **Research**: Exploring emergence and complexity
- 💻 **Development**: Adding features and optimizations
- 📚 **Documentation**: Improving explanations and tutorials
- 🧪 **Experimentation**: Testing new parameters and behaviors

**Get Started:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 💰 Support Development

Building transparent, educational AI takes time and resources. Support accelerates development, research, and community outreach.

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ff69b4)](https://github.com/sponsors/joeeddy)

**Your support enables:**
- Advanced research into emergent AI systems
- Educational materials and tutorials  
- Community workshops and events
- Open-source AI tools development

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

**Built with ❤️ for the future of transparent, decentralized intelligence.**