# PyTorch Graph Neural Networks Learning Repository

This repository contains educational implementations of Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric. The project demonstrates fundamental concepts in graph-based machine learning through simple, well-documented examples.

## 📁 Project Structure

### Core Files

#### 1. `simple_graph.py` - Graph Data Structure Foundation
This file demonstrates how to create and work with graph data structures using PyTorch Geometric.

**Key Components:**
- **Graph Definition**: Creates a simple 3-node graph with bidirectional connections
- **Node Features**: Each node has a single feature value ([-1], [0], [1])
- **Edge Connectivity**: Uses PyTorch Geometric's `edge_index` format
- **Graph Structure**: Linear chain topology (0 ↔ 1 ↔ 2)

**Graph Topology:**
```
Node 0 [-1] ←→ Node 1 [0] ←→ Node 2 [1]
```

**Technical Details:**
- Uses `torch_geometric.data.Data` class for graph representation
- Edge index format: `[[source_nodes], [target_nodes]]`
- Demonstrates key graph properties (num_nodes, num_edges, etc.)

#### 2. `recGNN.py` - Recurrent Graph Neural Network Implementation
This file implements a Recurrent Graph Neural Network (RecGNN) that performs iterative message passing.

**Algorithm Implementation:**
- **Recurrent GNN Update**: `H^(t+1) = tanh(W_self * H^(t) + W_neigh * A * H^(t))`
- **Message Passing**: 3 iterations of recurrent updates
- **Activation Function**: Uses `tanh` for non-linear transformation

**Key Components:**
- **Adjacency Matrix (A)**: 3×3 matrix representing graph connectivity
- **Hidden States (H)**: Node representations that evolve over time
- **Weight Parameters**:
  - `W_self` (0.6): Weight for current node's hidden state
  - `W_neigh` (0.4): Weight for aggregated neighbor information

**Algorithm Flow:**
1. **Neighbor Aggregation**: `neighbor_agg = A × H^(t)`
2. **State Update**: `H^(t+1) = tanh(W_self × H^(t) + W_neigh × neighbor_agg)`
3. **Iteration**: Repeat for 3 time steps

**Output Evolution:**
- Step 1: `[-0.537, 0.0, 0.537]`
- Step 2: `[-0.312, 0.0, 0.312]`
- Step 3: `[-0.185, 0.0, 0.185]`

The values converge towards zero, demonstrating the stabilizing effect of recurrent updates.

#### 3. `cGNN.py` - Convolutional Graph Neural Network Implementation
This file implements a Graph Convolutional Network (GCN) with multiple layers and different weights per layer.

**Algorithm Implementation:**
- **GCN Update**: `H^(l+1) = ReLU(A_norm × H^(l) × W^(l))`
- **Normalized Adjacency**: Uses degree normalization with self-loops
- **Multi-Layer Architecture**: 2 layers with different weight matrices

**Key Components:**
- **Normalized Adjacency Matrix (A_norm)**: `D^(-1/2) × (A + I) × D^(-1/2)`
- **Self-Loops Addition**: `A_hat = A + I` for better node representation
- **Layer-Specific Weights**:
  - `W1` (0.6): Weight matrix for first GCN layer
  - `W2` (0.4): Different weight matrix for second layer
- **ReLU Activation**: Non-linear activation between layers

**Algorithm Flow:**
1. **Normalization**: Compute normalized adjacency matrix with self-loops
2. **Layer 1**: `H1 = ReLU(A_norm × H × W1)`
3. **Layer 2**: `H2 = ReLU(A_norm × H1 × W2)`

**Output Evolution:**
- **Initial**: `[[-1.0], [0.0], [1.0]]`
- **After Layer 1**: `[[0.0], [0.0], [0.3]]`
- **After Layer 2**: `[[0.0], [0.049], [0.060]]`

#### 4. `CLAUDE.md` - Development Guide
Configuration file providing guidance for AI-assisted development in this repository.

**Contents:**
- Project overview and architecture
- Dependency management
- Development commands
- Code conventions and patterns

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.9.0+
- PyTorch Geometric 2.7.0+

### Installation
1. **Clone the repository**
2. **Activate virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torch-geometric numpy
   ```

### Running the Code

#### Basic Graph Exploration
```bash
python simple_graph.py
```
- Creates and displays a simple 3-node graph
- Demonstrates PyTorch Geometric data structures
- Shows graph properties and attributes

#### Recurrent GNN Training
```bash
python recGNN.py
```
- Runs 3 iterations of RecGNN updates
- Shows how node representations evolve
- Demonstrates message passing in action

## 🧠 Concepts Demonstrated

### Graph Neural Networks Fundamentals
- **Graph Representation**: How to encode graphs in PyTorch Geometric
- **Message Passing**: Information flow between connected nodes
- **Recurrent Updates**: Iterative refinement of node representations
- **Activation Functions**: Non-linear transformations in neural networks

### Mathematical Concepts
- **Adjacency Matrices**: Graph connectivity representation
- **Matrix Multiplication**: For neighbor aggregation
- **Recurrent Neural Networks**: Applied to graph-structured data
- **Parameter Learning**: Shared weights across time steps

## 📊 Graph Structure Details

### Node Information
| Node ID | Feature Value | Connections |
|---------|---------------|-------------|
| 0       | -1.0          | → 1         |
| 1       | 0.0           | → 0, 2      |
| 2       | 1.0           | → 1         |

### Edge Connections
- **Bidirectional edges**: Each connection works in both directions
- **Linear topology**: Forms a simple chain structure
- **No self-loops**: Nodes don't connect to themselves

## 🔬 Learning Objectives

This repository teaches:
1. **Graph Data Structures** in PyTorch Geometric
2. **Message Passing Algorithms** for information propagation
3. **Recurrent Neural Networks** applied to graphs
4. **Iterative Learning** and convergence behavior
5. **Parameter Sharing** in graph neural networks

## 🛠 Extending the Code

### Potential Enhancements
- Add more complex graph topologies
- Implement different GNN variants (GCN, GraphSAGE, GAT)
- Add node classification tasks
- Experiment with different activation functions
- Implement attention mechanisms

### Experimental Ideas
- Compare convergence rates with different weight values
- Test on larger graphs
- Add edge features
- Implement graph-level predictions

## 📚 References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
- [The Graph Neural Network Model](https://persagen.com/files/misc/scarselli2009graph.pdf)

---

**Note**: This is an educational repository focused on understanding GNN fundamentals through simple, clear implementations.