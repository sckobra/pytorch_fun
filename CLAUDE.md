# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch-based learning repository focused on Graph Neural Networks (GNNs) using PyTorch Geometric. The codebase contains implementations and experiments with graph-based machine learning models.

## Key Dependencies

- **PyTorch**: Core deep learning framework (torch==2.9.0)
- **PyTorch Geometric**: Graph neural network library (torch-geometric==2.7.0)
- **NumPy**: Numerical computing (numpy==2.3.4)

## Code Architecture

### Core Components

- **simple_graph.py**: Contains basic graph data structure definitions using PyTorch Geometric's `Data` class. Defines a simple 3-node graph with edge connections and node features.
- **recGNN.py**: Implements GNN computations including RecGNN updates with tanh activation and GCN layer updates with normalized adjacency matrices.

### Graph Structure

The codebase works with small graph examples:
- 3-node graphs with bidirectional edges
- Node features represented as tensors
- Edge connectivity defined using PyTorch Geometric's edge_index format

## Development Commands

### Running Code
```bash
python simple_graph.py    # Basic graph creation and exploration
python recGNN.py          # GNN model implementations and updates
```

### Environment Management
The project uses a virtual environment located in `.venv/`. Activate it before development:
```bash
# Windows
.venv\Scripts\activate

# Install dependencies (if needed)
pip install torch torch-geometric numpy
```

## Important Notes

- Graph data uses PyTorch Geometric's `Data` format with `x` (node features) and `edge_index` (connectivity)
- Edge indices follow PyTorch Geometric convention: `[source_nodes, target_nodes]`
- The codebase focuses on educational implementations of fundamental GNN concepts