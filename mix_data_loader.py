import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold

import networkx as nx
from torch_geometric.utils import to_networkx
from copy import deepcopy

from torch.utils.data import ConcatDataset
from tqdm import tqdm
datasets1 = [
    "AIDS",
    "DHFR",
    "BZR",
    "COX2",
    "PTC_MR",
    "MUTAG",
    "ENZYMES",
    "PROTEINS",
    "IMDB-MULTI",
    "IMDB-BINARY",
    "NCI1",
    "Tox21_HSE",
    "Tox21_MMP",
    "Tox21_p53",
    "Tox21_PPAR-gamma"
]
datasets2 = [
    "ogbg-molfreesolv",
    "ogbg-moltoxcast",
    "ogbg-molbbbp",
    "ogbg-molbace",
    "ogbg-moltox21",
    "ogbg-molsider",
    "ogbg-molclintox",
    "ogbg-mollipo",
    "ogbg-molesol",
    "ogbg-molmuv"
]

def calculate_graph_statistics(graphs):
    avg_nodes = 0
    avg_edges = 0

    for graph in graphs:
        G = to_networkx(graph, to_undirected=True)  # 转换为NetworkX图对象
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_nodes += num_nodes
        avg_edges += num_edges

    avg_nodes /= len(graphs)
    avg_edges /= len(graphs)

    return avg_nodes, avg_edges

def read_graph_file(DS, path):
    prefix = os.path.join(path, DS, DS)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]

        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1

        G_pyg = from_networkx(nx.relabel_nodes(G, mapping))
        G_pyg.y = G.graph['label']
        G_pyg.x = torch.ones((G_pyg.num_nodes,1))

        if G_pyg.num_nodes > 0:
            graphs.append(G_pyg)

    return graphs


def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)

    return gs

def merge_lists(lists, n):
    merged_list = []
    m = len(lists)
    remaining_n = n

    for i, sublist in enumerate(lists):
        remaining_lists = m - i
        target_length = remaining_n // remaining_lists

        if len(sublist) <= target_length:
            merged_list.extend(sublist)
            remaining_n -= len(sublist)
        else:
            merged_list.extend(sublist[:target_length])
            remaining_n -= target_length      
    return merged_list[:n]  

def get_all_realoe_dataloader(args, need_str_enc=True,max_l=-1):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    if DS in datasets1:
        mix_dataset_name_list = [d for d in datasets1 if d != DS and d != DS_ood]
    elif DS in datasets2:
        mix_dataset_name_list = [d for d in datasets2 if d != DS and d != DS_ood]
    else:
        assert False, f"DS: {DS}"
    
    all_datasets = []

    for one_dataset_name in mix_dataset_name_list:
        one_args = deepcopy(args)
        one_args.DS = one_dataset_name
        one_args.DS_ood = DS_ood
        one_args.DS_pair = None

        onedata_list_train = get_oe_dataset(one_args, train_per=1.0, need_str_enc=need_str_enc)


        all_datasets.append(onedata_list_train)
  

    if max_l==-1:
        combined_dataset = []
        for j in all_datasets:
            for i in j:
                i.edge_attr = None
                combined_dataset.append(i)
    else:
        all_datasets= sorted(all_datasets, key=len)
        combined_dataset = merge_lists(all_datasets, max_l)
        for i in combined_dataset:
            i.edge_attr = None
            # i.num_nodes =None
            i.num_nodes = i.x.size()[0]


    combined_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

    return combined_dataloader,combined_dataset,mix_dataset_name_list
    
def get_oe_dataset(args,max_l=-1,train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)

    if TU:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
        dataset_ood = TUDataset(path_ood, name=DS_ood, transform=(Constant(1, cat=False)))
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood, f"dataset_num_features: {dataset_num_features}, dataset_num_features_ood: {dataset_num_features_ood}"

    if max_l!=-1:
        dataset=dataset[:max_l]
    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]

    dataset_train = dataset[idx_train]

    data_list_train = []
    idx = 0

    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        # print(f'init structural encoding for {DS} : {len(data_list_train)}')
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    return data_list_train

def get_all_oe_dataset(args, need_str_enc=True,max_l=-1):

    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    if DS in datasets1:
        mix_dataset_name_list = [d for d in datasets1 if d != DS and d!=DS_ood]
    elif DS in datasets2:
        mix_dataset_name_list = [d for d in datasets2 if d != DS and d!=DS_ood]
    else:
        assert False, f"DS: {DS}"
    all_datasets = []

    for one_dataset in mix_dataset_name_list:
        one_args = deepcopy(args)
        one_args.DS = one_dataset
        one_args.DS_ood = DS_ood
        one_args.DS_pair = None

        onedata_list_train = get_oe_dataset(one_args,max_l=max_l,train_per=1.0, need_str_enc=need_str_enc)

        all_datasets.append(onedata_list_train)

    return all_datasets, mix_dataset_name_list

def get_ood_dataset_new(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
        args.DS=DS
        args.DS_ood=DS_ood
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)

    if TU:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
        dataset_ood = TUDataset(path_ood, name=DS_ood, transform=(Constant(1, cat=False)))
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood, f"dataset_num_features: {dataset_num_features}, dataset_num_features_ood: {dataset_num_features_ood}"

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train:])[0]

    dataset_train = dataset[idx_train]
    dataset_test = dataset[idx_test]
    dataset_ood = dataset_ood[: len(dataset_test)]

    data_list_train = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data.edge_attr = None
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        data.edge_attr = None
        data_list_test.append(data)

    if need_str_enc:
        data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    if train_per == 1.0:
        dataloader_test=None
    else:
        dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=True)

    # train和test都是id分布,test:ood=1:1
    meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
            'num_test':len(dataset_test), 'num_ood':len(dataset_ood)}

    return dataloader, dataloader_test, meta, [data_list_train,data_list_test,dataset_ood]


def get_ad_split_TU(args, fold=5):
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    
    dataset = TUDataset(path, name=DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits


def get_ad_dataset_TU(args, split, need_str_enc=True):
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    if DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=DS)

    dataset_num_features = dataset.num_node_features

    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())

    if need_str_enc:
        data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    data_train = []
    for data in data_train_:
        if data.y != 0:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 0 else 0

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train)}

    return dataloader, dataloader_test, meta


def get_ad_dataset_Tox21(args, need_str_enc=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')

    data_train_ = read_graph_file(args.DS + '_training', path)
    data_test = read_graph_file(args.DS + '_testing', path)

    dataset_num_features = data_train_[0].num_features

    data_train = []
    for data in data_train_:
        if data.y == 1:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 1 else 0

    if need_str_enc:
        data_train = init_structural_encoding(data_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
        data_test = init_structural_encoding(data_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train)}

    return dataloader, dataloader_test, meta