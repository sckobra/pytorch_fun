import torch

# Application Problem: 
# Given a small 3-node graph with adjacency matrix A,
# feature matrix X, and weights W, 
# compute one iteration of: 
# (a) RecGNN update using a tanh activation, 
# (b) GCN layer update with normalized adjacency

# Simple graph adjacency (3 nodes)
# 0 <-> 1 <-> 2
A = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float)

# Initial hidden states (h^(t))
H = torch.tensor([[-1.0], [0.0], [1.0]])  # shape (num_nodes, hidden_dim)

# Define parameters (shared weights)
W_self = torch.nn.Parameter(torch.tensor([[0.6]]))   # for current node
W_neigh = torch.nn.Parameter(torch.tensor([[0.4]]))  # for neighbors

# 1️⃣ AGGREGATE: sum (or mean) of neighbor hidden states
neighbor_agg = torch.matmul(A, H)  # shape (num_nodes, hidden_dim)

# 2️⃣ UPDATE: combine self + neighbor info
# Here we use a simple linear combination followed by tanh
H_new = torch.tanh(W_self * H + W_neigh * neighbor_agg)

print("Previous hidden states:\n", H)
print("Neighbor aggregation:\n", neighbor_agg)
print("Updated hidden states (H_new):\n", H_new)



