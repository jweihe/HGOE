
import faiss
import numpy as np
import torch
from torch_geometric.utils import dropout_node,dropout_edge,add_random_edge
##
def run_kmeans(x, args):

    results = {}

    d = x.shape[1]
    k = args.num_cluster
    clus = faiss.Clustering(d, k)
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 3

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False

    try:
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
    except:
        log.info('Fail to cluster with GPU. Try CPU...')
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 30),
                           np.percentile(density, 70))
    density = density / density.mean() + 0.5

    centroids = torch.Tensor(centroids).cuda()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster

    return results

def get_cluster_result(dataloader, model, args,use_oe=False):
    model.eval()
    if use_oe:
        b_all = torch.zeros((n_train_oe, model.embedding_dim))
    else:
        b_all = torch.zeros((n_train, model.embedding_dim))
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            b = model.get_b(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            b_all[data.idx] = b.detach().cpu()
    cluster_result = run_kmeans(b_all.numpy(), args)
    return cluster_result

def add_random_edges_to_batch(data_batch, edge_index,p=0.5, force_undirected=False):
    # 获取子图的ptr
    ptr = data_batch.ptr[:-1].cpu().numpy()

    # 初始化一个空的list来保存新的edge_index和添加的边
    new_edge_indices = []
    added_edge_indices = []

    # 对每个子图应用add_random_edge函数
    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i+1]
        edge_mask = (data_batch.batch[edge_index[0]] == i)
        subgraph_edge_index = edge_index[:, edge_mask] - start

        # 获取子图的节点数量
        num_nodes = end - start
        # 对子图应用函数
        new_edge_index, added_edge_index = add_random_edge(subgraph_edge_index, p=p, 
                                                           force_undirected=force_undirected,
                                                           num_nodes=num_nodes)
        # 将结果添加到list中
        new_edge_indices.append(new_edge_index + start)
        added_edge_indices.append(added_edge_index + start)
    # 合并所有子图的结果
    new_edge_index = torch.cat(new_edge_indices, dim=1)
    added_edges = torch.cat(added_edge_indices, dim=1)

    return new_edge_index, added_edges


def print_first_10_rows(data_batch):
    attributes = ['edge_index', 'x', 'edge_attr', 'y', 'idx', 'x_s', 'batch', 'ptr']

    for attr in attributes:
        attr_data = getattr(data_batch, attr, None)
        if attr_data is not None:
            print(attr)
            print(attr_data)

def modified_init_g_structural_encoding(g, edge_index, rw_dim=16, dg_dim=16):
    num_nodes = g.num_nodes
    edge_index = edge_index.long()
    
    # Construct adjacency matrix A
    A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1), device=edge_index.device), (num_nodes, num_nodes))
    
    # Calculate degree vector and avoid zero division by adding a small value
    if A._nnz() == 0:
        deg = torch.zeros(A.shape[0]).to(A.device) + 1e-10
    else:
        deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-10

    Dinv = torch.diag_embed(deg.pow(-1))
    
    # Calculate normalized adjacency matrix M
    M = torch.mm(A.to_dense(), Dinv)
    
    # Calculate RWSE
    RWSE = [M.diagonal()]
    M_power = M
    for _ in range(rw_dim - 1):
        M_power = torch.mm(M_power, M)
        RWSE.append(M_power.diagonal())
    RWSE = torch.stack(RWSE, dim=-1)
    
    # Calculate DGSE
    g_dg = deg.clip(0, dg_dim - 1).long()
    DGSE = torch.zeros([num_nodes, dg_dim], device=edge_index.device)
    DGSE.scatter_(1, g_dg.view(-1, 1), 1)
    
    # Concatenate RWSE and DGSE
    x_s = torch.cat([RWSE, DGSE], dim=1)
    
    return x_s


def init_g_structural_encoding(g,edge_index,rw_dim=16, dg_dim=16):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=g.num_nodes)
    D = (degree(edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    RWSE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(rw_dim-1):
        M_power = M_power * M
        RWSE.append(torch.from_numpy(M_power.diagonal()).float())
    RWSE = torch.stack(RWSE,dim=-1)

    g_dg = (degree(edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
    DGSE = torch.zeros([g.num_nodes, dg_dim])
    for i in range(len(g_dg)):
        DGSE[i, int(g_dg[i])] = 1

    x_s = torch.cat([RWSE, DGSE], dim=1)

    return x_s