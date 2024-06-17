import copy
import random
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import dropout_node,dropout_edge,add_random_edge
from utils.utils import modified_init_g_structural_encoding,add_random_edges_to_batch

def graph_random_aug(args,data,drop_edge_rate=0.0,add_edge_rate=0.0,drop_node_rate=0.0):
    # log.info('data',data)
    # log.info_first_10_rows(data)
    # drop node
    edge_index=data.edge_index
    if drop_node_rate!=0:
        edge_index, edge_mask, node_mask = dropout_node(edge_index,p=drop_node_rate)
    if drop_edge_rate!=0:
        edge_index, edge_mask = dropout_edge(edge_index, p=drop_edge_rate)
    if add_edge_rate!=0:
        edge_index, added_edges = add_random_edges_to_batch(data, edge_index, p=add_edge_rate)
    # new_x_s=init_g_structural_encoding(data,edge_index.cpu(),rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    new_x_s=modified_init_g_structural_encoding(data,edge_index,rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    # new_x_s=data.x_s
    return edge_index,new_x_s
    
def aug_random_edge(input_adj, drop_percent=0.2, add_percent=0.2):
    # Drop edges
    row_idx, col_idx = input_adj.nonzero()
    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)
    drop_num = int(edge_num * drop_percent*2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())
    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, drop_num)
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    # Add edges
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_num = int(edge_num * add_percent*2) 
    add_list = random.sample(l, add_num)
    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    # Convert and return
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj

if __name__=='__main__':
    # Testing the function
    adj_matrix = sp.csr_matrix([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    augmented_adj_matrix = aug_random_edge(adj_matrix, 0.2,0)

    print(augmented_adj_matrix.todense())
