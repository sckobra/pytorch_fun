import torch
# Application Problem: 
# Given a small 3-node graph with adjacency matrix A,
# feature matrix X, and weights W, 
# compute one iteration of: 
# (b) Convolutional GNN update

# Same simple 3-node graph
A = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float)

# Add self-loops for GCN later
I = torch.eye(A.size(0))
A_hat = A + I

# Compute the degree matrix and its inverse sqrt
D_hat = torch.diag(torch.sum(A_hat, dim=1))
D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))

# Normalized adjacency matrix
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

# Node feature matrix: one scalar feature per node
# Node 0: -1, Node 1: 0, Node 2: 1
H = torch.tensor([[-1.0], [0.0], [1.0]])

# GCN UPDATE (Convolutional GNN) -----------------------------

# Layer 1 weights (unique for this layer)
W1 = torch.tensor([[0.6]])
# Layer 2 weights (different from W1)
W2 = torch.tensor([[0.4]])


# ----- Layer 1 -----
# Propagate information with normalized adjacency
H1 = torch.relu(A_norm @ H @ W1)
print("=== GCN Layer 1 ===")
print("Input features:\n", H)
print("Normalized adjacency:\n", A_norm)
print("Output after layer 1:\n", H1, "\n")

# ----- Layer 2 -----
# Apply a second layer with a new weight
H2 = torch.relu(A_norm @ H1 @ W2)

print("Input features (layer 1 output):\n", H1)
print("Output after layer 2:\n", H2, "\n")


print("Final GCN node embeddings (after 2 layers):\n", H2)

# OUTPUT -------------------------------------------
# Input features:
#  tensor([[-1.],
#         [ 0.],
#         [ 1.]])
# Normalized adjacency:
#  tensor([[0.5000, 0.4082, 0.0000],
#         [0.4082, 0.3333, 0.4082],
#         [0.0000, 0.4082, 0.5000]])
# Output after layer 1:
#  tensor([[0.0000],
#         [0.0000],
#         [0.3000]])

# === GCN Layer 2 ===
# Input features (layer 1 output):
#  tensor([[0.0000],
#         [0.0000],
#         [0.3000]])
# Output after layer 2:
#  tensor([[0.0000],
#         [0.0490],
#         [0.0600]])

# Final GCN node embeddings (after 2 layers):
#  tensor([[0.0000],
#         [0.0490],
#         [0.0600]])