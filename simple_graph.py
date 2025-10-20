import torch
from torch_geometric.data import Data

# SIMPLE GRAPH ----------------------

# defining edges (0 -> 1, 1 -> 0, 1 -> 2, 2 -> 1) first array is the source nodes, they correspond to the target nodes in the second array
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], 
                           dtype=torch.long)

# defining the node feature vectors: in this case, 1 feature per node
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

mySimpleGraph = Data(x=x, edge_index=edge_index)
# print(mySimpleGraph)

# OUTPUT -----------------------------
# >>> Data(x=[3, 1], edge_index=[2, 4])
# edge_index is the shape of the edge matrix
# x = [3, 1] 3 nodes, 1 feature per node


# Useful attributes & functions:

# print(data.num_nodes)

# print(data.num_edges)

# print(data.num_node_features)

# print(data.has_isolated_nodes())

