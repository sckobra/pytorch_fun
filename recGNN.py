# Application Problem: 
# Given a small 3-node graph with adjacency matrix A,
# feature matrix X, and weights W, 
# compute one iteration of: 
# (a) RecGNN update using a tanh activation, 
# (b) GCN layer update with normalized adjacency

import torch
import simple_graph
from torch_geometric.data import Data

print(simple_graph.mySimpleGraph)

