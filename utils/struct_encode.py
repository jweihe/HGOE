import numpy as np
import scipy.sparse as sp
import torch

# Original function
def original_init_g_structural_encoding(g, edge_index, rw_dim=16, dg_dim=16):
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index.cpu().numpy()), shape=(g.num_nodes, g.num_nodes)).tocsc()
    D = (np.asarray(A.sum(1)).flatten() ** -1.0)

    Dinv = sp.diags(D)
    RW = A.dot(Dinv)
    M = RW

    RWSE = [torch.from_numpy(np.array(M.diagonal())).float().cuda()]
    M_power = M
    for _ in range(rw_dim - 1):
        M_power = M_power.dot(M)
        RWSE.append(torch.from_numpy(np.array(M_power.diagonal())).float().cuda())
    RWSE = torch.stack(RWSE, dim=-1)

    g_dg = (np.asarray(A.sum(1)).flatten()).clip(0, dg_dim - 1)
    DGSE = torch.zeros([g.num_nodes, dg_dim], device=edge_index.device)
    for i in range(len(g_dg)):
        DGSE[i, int(g_dg[i])] = 1

    x_s = torch.cat([RWSE, DGSE], dim=1)

    return x_s



def modified_init_g_structural_encoding(g, edge_index, rw_dim=16, dg_dim=16):
    edge_index = edge_index.long().to(edge_index.device)
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=edge_index.device), (g.num_nodes, g.num_nodes)).coalesce().to_dense()
    D = torch.sum(A, dim=1).flatten() ** -1.0
    Dinv = torch.diag_embed(D)

    # Calculate normalized adjacency matrix M
    M = torch.mm(A, Dinv)

    # Calculate RWSE
    RWSE = [M.diagonal()]
    M_power = M
    for _ in range(rw_dim - 1):
        M_power = torch.mm(M_power, M)
        RWSE.append(M_power.diagonal())
    RWSE = torch.stack(RWSE, dim=-1)

    # Calculate DGSE
    g_dg = torch.clamp(torch.sum(A, dim=1).flatten(), 0, dg_dim - 1).long()
    DGSE = torch.zeros([g.num_nodes, dg_dim], device=edge_index.device)
    DGSE.scatter_(1, g_dg.view(-1, 1), 1)

    # Concatenate RWSE and DGSE
    x_s = torch.cat([RWSE, DGSE], dim=1)

    return x_s
#0.8s
# def modified_init_g_structural_encoding(g, edge_index, rw_dim=16, dg_dim=16):
    # num_nodes = g.num_nodes
    # edge_index = edge_index.long()
    
    # # Construct adjacency matrix A
    # A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1), device=edge_index.device), (num_nodes, num_nodes))
    
    # # Calculate degree vector and avoid zero division by adding a small value
    # if A._nnz() == 0:
    #     deg = torch.zeros(A.shape[0]).to(A.device) + 1e-10
    # else:
    #     deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-10

    # Dinv = torch.diag_embed(deg.pow(-1))
    
    # # Calculate normalized adjacency matrix M
    # M = torch.mm(A.to_dense(), Dinv)
    
    # # Calculate RWSE
    # RWSE = [M.diagonal()]
    # M_power = M
    # for _ in range(rw_dim - 1):
    #     M_power = torch.mm(M_power, M)
    #     RWSE.append(M_power.diagonal())
    # RWSE = torch.stack(RWSE, dim=-1)
    
    # # Calculate DGSE
    # g_dg = deg.clip(0, dg_dim - 1).long()
    # DGSE = torch.zeros([num_nodes, dg_dim], device=edge_index.device)
    # DGSE.scatter_(1, g_dg.view(-1, 1), 1)
    
    # # Concatenate RWSE and DGSE
    # x_s = torch.cat([RWSE, DGSE], dim=1)
    
    # return x_s

# Test cases
def test_functions_equivalence():
    class Graph:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes

    # Test case: Simple graph with 5 nodes and few edges
    g = Graph(5)
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 4], 
                               [1, 2, 3, 4, 4, 3]], device='cuda')

    result_modified = modified_init_g_structural_encoding(g, edge_index) 
    result_original = original_init_g_structural_encoding(g, edge_index)


    print('result_original',result_original)
    print('result_modified',result_modified)
    # Checking if results are approximately similar (due to potential floating-point differences)
    are_equivalent = torch.allclose(result_original, result_modified, atol=1e-5)
    
    return are_equivalent

# # Run the test
# print(test_functions_equivalence())
import time

# def optimized_init_g_structural_encoding(g, edge_index, rw_dim=16, dg_dim=16):
#     num_nodes = g.num_nodes
#     edge_index = edge_index.long()
    
#     # Construct adjacency matrix A
#     A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1), device=edge_index.device), (num_nodes, num_nodes))
    
#     # Calculate degree vector and avoid zero division by adding a small value
#     deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-10
#     Dinv = torch.diag_embed(deg.pow(-1))
    
#     # Calculate normalized adjacency matrix M
#     M = torch.mm(A.to_dense(), Dinv)
    
#     # Calculate RWSE
#     RWSE = [M.diagonal()]
#     M_power = M
#     for _ in range(rw_dim - 1):
#         M_power = torch.mm(M_power, M)
#         RWSE.append(M_power.diagonal())
#     RWSE = torch.stack(RWSE, dim=-1)
    
#     # Calculate DGSE
#     g_dg = deg.clip(0, dg_dim - 1).long()
#     DGSE = torch.zeros([num_nodes, dg_dim], device=edge_index.device)
#     DGSE.scatter_(1, g_dg.view(-1, 1), 1)
    
#     # Concatenate RWSE and DGSE
#     x_s = torch.cat([RWSE, DGSE], dim=1)
    
#     return x_s
def measure_performance(func, *args, **kwargs):
    # Record the initial GPU memory
    start_mem = torch.cuda.memory_allocated()
    
    # Record the starting time
    start_time = time.time()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Record the ending time
    end_time = time.time()
    
    # Record the final GPU memory
    end_mem = torch.cuda.memory_allocated()
    
    # Calculate the differences
    runtime = end_time - start_time
    mem_usage = end_mem - start_mem
    
    return runtime, mem_usage, result

# Create a test graph
class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

g = Graph(5)
edge_index = torch.tensor([[0, 1, 2, 3, 1, 4], 
                           [1, 2, 3, 4, 4, 3]], device='cuda')

# Measure performance for both functions
modified_time, modified_mem, _ = measure_performance(modified_init_g_structural_encoding, g, edge_index)
original_time, original_mem, _ = measure_performance(original_init_g_structural_encoding, g, edge_index)


print(test_functions_equivalence())
# original_time, original_mem, modified_time, modified_mem
print('original_time, original_mem, modified_time, modified_mem',original_time, original_mem, modified_time, modified_mem)
