from typing import List, Tuple
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import copy
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_dense_adj
import torch.nn.functional as F
import torch
import random

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from mix_data_loader import init_structural_encoding
from torch_geometric.loader import DataLoader
from tqdm import tqdm

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        # print( data.x.shape )
        return data


def prepare_synthetic_dataset(dataset):
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)

            data.x = F.one_hot(degs.to(torch.int64), num_classes=max_degree+1).to(torch.float)
            print(data.x.shape)


        return dataset


def prepare_dataset(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, dim=1)

# def prepare_dataloader_x_xs(dataset,dataset_for_search,args):
#     print('start prepare_dataloader_x_xs')
#     data_list= []
#     idx=0
#     for data in dataset:
#         data.edge_attr = None
#         data['idx'] = idx
#         idx += 1
#         data_list.append(data)

#     data_list= init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
#     # Determine the maximum number of nodes (rows in x_s) across all data in both dataset and dataset_for_search
#     max_nodes = max(max(data.x_s.size(0) for data in dataset), max(data.x_s.size(0) for data in dataset_for_search))
    
#     # Concatenate all x_s from dataset_for_search to form a single large tensor
#     all_x_s_search = torch.cat([node.x_s for node in dataset_for_search], dim=0)
#     all_x_search = torch.cat([node.x for node in dataset_for_search], dim=0)  # Concatenate all x from dataset_for_search
#     for data in tqdm(data_list):
#         distances = torch.cdist(data.x_s, all_x_s_search)
#         closest_indices = distances.argmin(dim=1)
        
#         # Assign closest x values from dataset_for_search to data's x attribute
#         data.x = all_x_search[closest_indices]

#     dataloader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

#     return dataloader,data_list

def prepare_dataloader_x_xs(dataset, dataset_for_search,close_k,args):
    print('start prepare_dataloader_x_xs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_list = []
    idx = 0
    for data in dataset:
        data.edge_attr = None
        data['idx'] = idx
        idx += 1
        data_list.append(data)

    data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    
    # Determine the maximum number of nodes (rows in x_s) across all data in both dataset and dataset_for_search
    # max_nodes = max(max(data.x_s.size(0) for data in dataset), max(data.x_s.size(0) for data in dataset_for_search))
    
    # Concatenate all x_s from dataset_for_search to form a single large tensor
    all_x_s_search = torch.cat([node.x_s for node in dataset_for_search], dim=0).to(device)
    all_x_search = torch.cat([node.x for node in dataset_for_search], dim=0).to(device)  # Concatenate all x from dataset_for_search
    
    for data in tqdm(data_list):
        if data.x is None:
            distances = torch.cdist(data.x_s.to(device), all_x_s_search)
            closest_indices = distances.argmin(dim=1)
            
            print('closest_indices',closest_indices.size())
            # Assign closest x values from dataset_for_search to data's x attribute
            data.x = all_x_search[closest_indices]

    dataloader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)

    return dataloader, data_list

def prepare_dataset_x_xs_mean(dataset, dataset_for_search, close_k, args):
    print('start prepare_dataloader_x_xs mean')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_list = []
    idx = 0
    for data in dataset:
        data.edge_attr = None
        data['idx'] = idx
        idx += 1
        data_list.append(data)

    data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    all_x_s_search = torch.cat([node.x_s for node in dataset_for_search], dim=0).to(device)
    all_x_search = torch.cat([node.x for node in dataset_for_search], dim=0).to(device)

    for data in tqdm(data_list):
        if data.x is None:
            distances = torch.cdist(data.x_s.to(device), all_x_s_search)
            closest_indices = distances.topk(k=close_k, largest=False)[1]
            x = torch.mean(all_x_search[closest_indices], dim=1)
            data.x=x.detach().cpu()

            # distances = torch.cdist(data.x_s.to(device), all_x_s_search)
            # closest_indices = distances.argmin(dim=1)
            # # Assign closest x values from dataset_for_search to data's x attribute
            # data.x = all_x_search[closest_indices]

    # dataloader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)
    return data_list

def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()



def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []

    normalized_node_degrees = []
    for i in tqdm(range(len(graphs))):
    # for i in tqdm(range(len(graphs))):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num

def align_tensor_graphs(graphs: List[torch.Tensor],
                              padding: bool = False, N: int = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []

    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * torch.sum(graphs[i], dim=0) + 0.5 * torch.sum(graphs[i], dim=1)
        node_degree /= torch.sum(node_degree)
        idx = torch.argsort(node_degree, descending=True)  # descending

        sorted_node_degree = node_degree[idx].unsqueeze(-1)

        sorted_graph = graphs[i][idx, :]
        sorted_graph = sorted_graph[:, idx]

        if N:
            max_num = max(max_num, N)

        if padding:
            normalized_node_degree = torch.zeros((max_num, 1), device=device)
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = torch.zeros((max_num, max_num), device=device)
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

    if N:
        aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
        normalized_node_degrees = [node_degree[:N] for node_degree in normalized_node_degrees]

    return aligned_graphs, normalized_node_degrees, max_num, min_num

def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        node_x = copy.deepcopy( node_x )
        sorted_node_x = node_x[ idx, :]

        max_num = max(max_num, N)
        # if max_num < N:
        #     max_num = max(max_num, N)
        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            # added
            aligned_node_x = np.zeros((max_num, 1))
            aligned_node_x[:num_i, :] = sorted_node_x


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            #added
            aligned_node_x = aligned_node_x[:N]

    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num





def two_graphons_mixup(two_graphons, la=0.5, num_sample=20):

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        # print(edge_index)
    return sample_graphs

def adjust_graphon_size(graphon: np.ndarray, N: int) -> np.ndarray:
    """
    Adjust the size of a given graphon to N x N.

    :param graphon: input graphon (2D numpy array)
    :param N: desired size
    :return: adjusted graphon
    """
    current_size = graphon.shape[0]
    
    # If graphon size is less than N
    if current_size < N:
        # Create an N x N matrix filled with zeros
        new_graphon = np.zeros((N, N))
        # Place the original graphon in the top-left corner
        new_graphon[:current_size, :current_size] = graphon
        return new_graphon
    
    # If graphon size is greater than N
    elif current_size > N:
        # Return the top-left N x N sub-matrix
        return graphon[:N, :N]
    
    # If graphon size is already N
    else:
        return graphon
        
def two_graphon_mixup(graphon1,graphon2,la=0.5, num_sample=20):

    # label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * graphon1 + (1 - la) * graphon2
    label = la * 1 + (1 - la) * 0

    # sample_graph_label = torch.from_numpy(label)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        if edge_index.numel() == 0:
            print("edge_index is empty")
            i=i-1
            continue
        else:
            num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        # print(edge_index)
    return sample_graphs

def two_graphon_mixup_random_align(graphon1,graphon2,min_size,max_size,la=0.5, num_sample=20):

    # label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    # new_graphon = la * graphon1 + (1 - la) * graphon2
    label = la * 1 + (1 - la) * 0

    # sample_graph_label = torch.from_numpy(label)
    # print(new_graphon)

    sample_graphs = []

    num_s=0
    while num_s <num_sample:
        choice_size=random.randint(int(min_size),int(max_size))

        adjust_graphon1=adjust_graphon_size(graphon1,choice_size)
        adjust_graphon2=adjust_graphon_size(graphon2,choice_size)
        # new_graphon = la * graphon1 + (1 - la) * graphon2
        new_graphon = la * adjust_graphon1 + (1 - la) * adjust_graphon2
        # new_graphon=new_graphon + np.mean(new_graphon)
        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        if edge_index.numel() == 0:
            print("edge_index is empty")
            continue
        else:
            num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        num_s+=1
        # print(edge_index)
    return sample_graphs


def two_x_graphons_mixup(two_x_graphons, la=0.5, num_sample=20):

    label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
    new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
    new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    sample_graph_x = torch.from_numpy(new_x).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.x = sample_graph_x
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        # print(edge_index)
    return sample_graphs



def graphon_mixup(dataset, la=0.5, num_sample=20):
    graphons = estimate_graphon(dataset, universal_svd)

    two_graphons = random.sample(graphons, 2)
    # for label, graphon in two_graphons:
    #     print( label, graphon )
    # print(two_graphons[0][0])

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    print("new label:", label)
    # print("new graphon:", new_graphon)

    # print( label )
    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) < new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]

        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        # print(sample_graph.shape)

        # print(sample_graph)

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)
        # print(edge_index)
    return sample_graphs


def estimate_graphon(dataset, graphon_estimator):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    # print(len(all_graphs_list))

    graphons = []
    for class_label in set(y_list):
        c_graph_list = [ all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label ]

        aligned_adj_list, normalized_node_degrees, max_num, min_num = align_graphs(c_graph_list, padding=True, N=400)

        graphon_c = graphon_estimator(aligned_adj_list, threshold=0.2)

        graphons.append((np.array(class_label), graphon_c))

    return graphons



def estimate_one_graphon(aligned_adj_list: List[np.ndarray], method="universal_svd"):

    if method == "universal_svd":
        graphon = universal_svd(aligned_adj_list, threshold=0.2)
    else:
        graphon = universal_svd(aligned_adj_list, threshold=0.2)

    return graphon



def split_class_x_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    all_node_x_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)
        all_node_x_list = [graph.x.numpy()]

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list, c_node_x_list ) )

    return class_graphs


def split_class_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list ) )

    return class_graphs

def split_graphs(dataset):

    y_list = []
    print('start split_graphs')
    all_graphs_list=[]
    for graph in tqdm(dataset):
        # adj = to_dense_adj(graph.edge_index)[0].numpy()
        # adj = to_dense_adj(graph.edge_index)[0].cuda()
        adj = to_dense_adj(graph.edge_index)[0]
        all_graphs_list.append(adj)

    return all_graphs_list

def split_dataset_by_node_count(dataset, k):
    # Step 1: 对图按照节点数量进行排序
    sorted_dataset = sorted(dataset, key=lambda graph: graph.num_nodes)
    
    # Step 2: 将图均匀地分配到 k 个新的数据集中
    # 计算每个数据集应该有的图的数量
    avg_len = len(sorted_dataset) // k
    datasets = []
    
    for i in range(k):
        start_idx = i * avg_len
        if i == k-1:  # 最后一个数据集获取所有剩余的图
            end_idx = len(sorted_dataset)
        else:
            end_idx = start_idx + avg_len

        datasets.append(sorted_dataset[start_idx:end_idx])
    

    return datasets

def count_nodes_distribution(dataset):
    # 字典用于统计节点数量
    nodes_count_dict = {}
    
    for graph in dataset:
        # 计算每个图的节点数量
        num_nodes = graph.num_nodes
        
        # 更新字典中的计数
        if num_nodes in nodes_count_dict:
            nodes_count_dict[num_nodes] += 1
        else:
            nodes_count_dict[num_nodes] = 1
            
    # 返回排序后的字典
    return dict(sorted(nodes_count_dict.items()))

def split_graphs_gpu_optimized(dataset):
    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].cuda()
        all_graphs_list.append(adj)
    return all_graphs_list


def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)


    u, s, v = torch.svd(sum_graph)

    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0

    graphon = u @ torch.diag(s) @ torch.t(v)

    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    # torch.cuda.empty_cache()

    return graphon

def universal_tensor_svd(aligned_graphs: List[torch.tensor], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs =torch.stack(aligned_graphs, dim=0).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    # torch.cuda.empty_cache()
    return graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method

    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon


# def stat_graph(graphs_list: List[Data]):
#     num_total_nodes = []
#     num_total_edges = []
#     for graph in graphs_list:
#         num_total_nodes.append(graph.num_nodes)
#         num_total_edges.append(  graph.edge_index.shape[1] )
#     avg_num_nodes = sum( num_total_nodes ) / len(graphs_list)
#     avg_num_edges = sum( num_total_edges ) / len(graphs_list) / 2.0
#     avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

#     median_num_nodes = np.median( num_total_nodes ) 
#     median_num_edges = np.median(num_total_edges)
#     median_density = median_num_edges / (median_num_nodes * median_num_nodes)

#     return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density

def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(graph.edge_index.shape[1])
        
    avg_num_nodes = sum(num_total_nodes) / len(graphs_list)
    avg_num_edges = sum(num_total_edges) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median(num_total_nodes)
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    min_num_nodes = min(num_total_nodes)
    max_num_nodes = max(num_total_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, max_num_nodes
