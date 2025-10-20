import torch

# Application Problem: 
# Given a small 3-node graph with adjacency matrix A,
# feature matrix X, and weights W, 
# compute one iteration of: 
# (a) RecGNN update using a tanh activation 3 times

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

for t in range(3):  # 3 recurrent steps
    neighbor_agg = torch.matmul(A, H)
    H = torch.tanh(W_self * H + W_neigh * neighbor_agg)
    print(f"After step {t+1}: {H.squeeze().tolist()}")

# OUTPUT -----------------------------------------------------------
# After step 1: [-0.5370495915412903, 0.0, 0.5370495915412903]
# After step 2: [-0.3115217089653015, 0.0, 0.3115217089653015]
# After step 3: [-0.18476632237434387, 0.0, 0.18476632237434387]