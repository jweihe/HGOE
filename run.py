def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import os

from model import HCL
from data_loader import *
from mix_data_loader import get_all_realoe_dataloader,get_ood_dataset_new,get_oe_dataset
import argparse
import numpy as np
import torch
import random
import faiss
import sklearn.metrics as skm
import torch_geometric
import copy
import datetime
import csv
import pytz

import logging
import time
import math

import torch.nn.functional as F
from torch_geometric.utils import dropout_node,dropout_edge,add_random_edge
##
# import cProfile

from utils.graphon_utils import stat_graph, split_class_graphs, align_graphs,align_tensor_graphs
from utils.graphon_utils import two_graphons_mixup, universal_svd,universal_tensor_svd
from utils.graphon_utils import split_graphs,two_graphon_mixup,prepare_dataloader_x_xs,split_graphs_gpu_optimized


from utils.graph_aug import graph_random_aug

from utils.loss import entropy_loss,baloss
from utils.utils import run_kmeans,get_cluster_result,add_random_edges_to_batch,print_first_10_rows,modified_init_g_structural_encoding,init_g_structural_encoding

from utils.graphon_utils import count_nodes_distribution,split_dataset_by_node_count,adjust_graphon_size,two_graphon_mixup_random_align

from mix_data_loader import get_all_oe_dataset

from graph_ssl.GraphCL import GraphCLModel
from graph_ssl.cluster import split_dataset_by_ssl_feature
from utils.graphon_utils import prepare_dataset_x_xs_mean
from torch_geometric.loader import DataLoader

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
    "Tox21_p51",
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')

    parser.add_argument('-DS_oe', help='Dataset', default='None')

    parser.add_argument('-OE', help='0: origin without oe, 1: real OE, 2:aug OE, 1:multi source OE', type=int,default=0)

    parser.add_argument('-realoe_rate', type=float, default=0,help='use real OE rate')
    parser.add_argument('-mixup_rate', type=float, default=0,help='use mixup OE rate')
    parser.add_argument('-inter_rate', type=float, default=0,help='use inter OE rate')

    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-oe_num_cluster', type=int, default=10)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-oe_weight', type=float, default=1)  

    # for aug oe
    parser.add_argument('-edge_drop_r', type=float, default=0.1)  
    parser.add_argument('-node_drop_r', type=float, default=0.1)  
    parser.add_argument('-edge_add_r', type=float, default=0.1)  

    parser.add_argument('-oe_loss_type', type=str, choices=['origin','baloss'], default='log')
    parser.add_argument('-log_epsilon', type=float, default=10)
    parser.add_argument('-pairlog_epsilon', type=float, default=1)
    
    parser.add_argument('-gamma', type=float, default=1.0)
    parser.add_argument('-margin', type=float, default=1.0)

    parser.add_argument('-atom_size_rate', type=float, default=0.1)
    parser.add_argument('-atom_q', type=float, default=0.0)

    # parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('-lam_range', type=str, default="[0.7,1.0]")
    parser.add_argument('-aug_ratio', type=float, default=0.15)
    parser.add_argument('-aug_num', type=int, default=10)

    parser.add_argument('-id_cluster_num', type=int, default=0)
    parser.add_argument('-keep_oe_graphon_size', type=int, default=0)
    parser.add_argument('-cluster_type',type=str,default='graphCL',choices=['node','graphCL','infograph'])
    parser.add_argument('-ssl_epoch', type=int, default=100)
    parser.add_argument('-ssl', type=bool, default=False)
    parser.add_argument('-search_all', type=bool, default=False)

    parser.add_argument('-select_sample', type=int, default=0,help='0:do not select sample,1:select sample by ssl feature,2:selsect sample by structural feature')
    parser.add_argument('-close_k', type=int, default=1)
    

    return parser.parse_args()

def get_time():

    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)
    return utc_now

def setup_logger(args):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    lam_range = eval(args.lam_range)

    if args.OE==0:
        filepath = f'good'
    elif args.OE==1:
        filepath = f'oe_real{args.realoe_rate}-mix{args.mixup_rate}-inter{args.inter_rate}_{args.DS_pair}_w_{args.oe_weight}_{args.oe_loss_type}'
    if args.id_cluster_num>1:
        filepath += f'_{args.cluster_type}_{args.id_cluster_num}'
        
    log_file = f"{filepath}_{get_time().strftime('%m%d-%H%M')}.log"
    log_file = os.path.join(log_dir, log_file)

    def normal(sec, what):
        normal_time = datetime.datetime.now() + datetime.timedelta(hours=12)
        return normal_time.timetuple()

    logging.Formatter.converter = normal
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger()

if __name__ == '__main__':
    setup_seed(0)
    args = arg_parse()
    log = setup_logger(args)
    num_epoch=args.num_epoch

    # adjust oe mode depend on args.OE
    if args.OE==1:
        if not args.realoe_rate and not args.mixup_rate and not args.inter_rate:
            assert False,'please choose oe mode when using multi oe'
    else:
        args.realoe_rate=0
        args.mixup_rate=0
        args.inter_rate=0

    log.info(args)
    if args.id_cluster_num>1:
        if args.cluster_type=='node':
            log.info(f'cluster: {args.cluster_type}')
        else:
            args.ssl=True
            log.info(f'cluster: {args.cluster_type}')

    root_path = '.'

    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    if args.OE==1:
        oe_args = copy.deepcopy(args)
        oe_args.DS_pair = None
        oe_args.DS = args.DS_oe
        oe_args.DS_ood = args.DS_oe
        oe_args.num_cluster = args.oe_num_cluster

    aucs = []
    for trial in range(args.num_trial):
        log.info(f'Trial {trial}')
        setup_seed(trial + 1)

        # set oe dataset
        
        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta, dataset_triple = get_ood_dataset_new(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial])

        if args.ssl:
            if args.cluster_type=='graphCL':
                ssl_model = GraphCLModel(dataloader,num_feature=meta['num_feat'],TU = not args.DS.startswith('ogbg-mol'))
                ssl_model.train_model(args.ssl_epoch)
                
        log.info(f'meta: {meta}')

        if args.OE==1:
            if args.mixup_rate or args.inter_rate:
                avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density,min_num_nodes, max_num_nodes = stat_graph(dataset_triple[0])


                # resolution = int(median_num_nodes+max_num_nodes)//2
                # resolution=min(int(median_num_nodes)*2,resolution)

                resolution=int(median_num_nodes)

                mixoe_dataset,mixoe_dataset_name_list= get_all_oe_dataset(args, max_l=int(meta['num_train']*args.mixup_rate))
                if args.search_all:
                    mixoe_dataset_all=[iii for ii in mixoe_dataset for iii in ii]
                id_graphon_list=[]
                if args.id_cluster_num>1:
                    id_graphon_name_list=[]
                    for ii in range(args.id_cluster_num):
                        if args.cluster_type=='node':
                            id_graphon_name_list.append(f'ID_{args.DS}_{resolution}_node-count-cluster-{args.id_cluster_num}_{ii}.npy')
                        elif args.cluster_type=='graphCL':
                            id_graphon_name_list.append(f'ID_{args.DS}_{resolution}_graphCL-cluster-{args.id_cluster_num}_{ii}.npy')

                    id_graphon_path_list=[os.path.join(root_path,'graphon_lib',id_graphon_name) for id_graphon_name in id_graphon_name_list]

                    if os.path.exists(id_graphon_path_list[-1]):
                        for id_graphon_path in id_graphon_path_list:
                            id_graphon=np.load(id_graphon_path)
                            id_graphon_list.append(id_graphon)
                    else: 
                        if  args.cluster_type=='node':
                            k_dataset=split_dataset_by_node_count(dataset_triple[0],args.id_cluster_num)
                        elif args.cluster_type=='graphCL':
                            k_dataset,im2cluster,centroids,density=split_dataset_by_ssl_feature(args,model=ssl_model,dataset=dataset_triple[0],dataloader=dataloader,n_train=meta['num_train'],device='cuda')

                        for data_idx,one_dataset in enumerate(k_dataset):
                            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, max_num_nodes = stat_graph(one_dataset)
                    
                            one_resolution=int(median_num_nodes)
              
                            graphs = split_graphs(one_dataset)
                

                            align_graphs_list, normalized_node_degrees, max_num, min_num = align_tensor_graphs(graphs, padding=True, N=one_resolution)

                            one_id_graphon = universal_tensor_svd(align_graphs_list, threshold=0.2)
                            id_graphon_list.append(one_id_graphon)
                            np.save(id_graphon_path_list[data_idx],one_id_graphon)

                else:
                    id_graphon_name=f'ID_{args.DS}_{resolution}.npy'
                    id_graphon_path=os.path.join(root_path,'graphon_lib',id_graphon_name)
                    if os.path.exists(id_graphon_path):
                        id_graphon=np.load(id_graphon_path)
                        log.info(f'load id graphon from {id_graphon_path}')
                    else:
                        graphs = split_graphs(dataset_triple[0])
    
                        align_graphs_list, normalized_node_degrees, max_num, min_num = align_tensor_graphs(graphs, padding=True, N=resolution)
                        id_graphon = universal_tensor_svd(align_graphs_list, threshold=0.01)
                        np.save(id_graphon_path,id_graphon)
                    id_graphon_list.append(id_graphon)

            start_time = time.time()
            
            if args.mixup_rate:

                mixoe_graphon_list=[]
                mid_oe_size_list=[]
                max_oe_size_list=[] 
                for idx,one_oe_dataset in enumerate(mixoe_dataset):
                    avg_num_nodes_oe, avg_num_edges_oe, avg_density_oe, median_num_nodes_oe, median_num_edges_oe, median_density_oe,min_num_nodes_oe, max_num_nodes_oe = stat_graph(one_oe_dataset)
                    mid_oe_size_list.append(median_num_nodes_oe)
                    max_oe_size_list.append(max_num_nodes_oe)

                    one_oe_resolution=int(median_num_nodes_oe)

        
                    one_oe_dataset_name=mixoe_dataset_name_list[idx]
                    one_oe_graphon_name=f'OE_{one_oe_dataset_name}_{one_oe_resolution}.npy'
                    one_oe_graphon_path=os.path.join(root_path,'graphon_lib',one_oe_graphon_name)
                    
                    if os.path.exists(one_oe_graphon_path):
                        one_oe_graphon=np.load(one_oe_graphon_path)
                    else:
                        graphs_mixoe=split_graphs(one_oe_dataset)
                        align_graphs_list_oe, normalized_node_degrees_oe, max_num_oe, min_num_oe = align_tensor_graphs(graphs_mixoe, padding=True, N=one_oe_resolution)
                        # log.info('align_graphs_list_oe',align_graphs_list_oe)
                        # log.info(f"aligned graph {align_graphs_list_oe[0].shape}" )
                        one_oe_graphon = universal_tensor_svd(align_graphs_list_oe, threshold=0.01)  
                        np.save(one_oe_graphon_path,one_oe_graphon)
                    mixoe_graphon_list.append(one_oe_graphon)


        if args.OE ==1:

            lam_range = eval(args.lam_range)

            lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(args.aug_num,))
            if args.realoe_rate:
                dataloader_realoe, realoe_dataset,mix_dataset_name_list= get_all_realoe_dataloader(args, max_l=int(meta['num_train']*args.realoe_rate))

            if args.mixup_rate:
                new_graph=[]
                num_sample=math.ceil(meta['num_train']*args.mixup_rate / (args.aug_num * len(mixoe_graphon_list)*len(id_graphon_list)))
                for id_graphon in id_graphon_list:
                    for oe_idx,oe_graphon in enumerate(mixoe_graphon_list):
                        oe_graph_show=[]
                        if not args.keep_oe_graphon_size:
                            min_size=2
                            max_size=max(id_graphon.shape[0],oe_graphon.shape[0])
                        for lam in lam_list:
                            if args.keep_oe_graphon_size:
                                adjust_id_graphon=adjust_graphon_size(id_graphon,oe_graphon.shape[0])
                                oe_graph_new=two_graphon_mixup(adjust_id_graphon,oe_graphon, la=lam, num_sample=num_sample)
                                oe_graph_show+=oe_graph_new
                                new_graph += oe_graph_new
                            else: 
                                oe_graph_new=two_graphon_mixup_random_align(id_graphon,oe_graphon,min_size=min_size,max_size=max_size,la=lam, num_sample=num_sample)
                                oe_graph_show+=oe_graph_new
                                new_graph+=oe_graph_new
                mixup_dataset=new_graph



            if args.inter_rate:
                new_graph=[]
                id_inter_sample_times=(len(id_graphon_list)^2 //2)
                num_sample=math.ceil(meta['num_train']*args.inter_rate / (args.aug_num * id_inter_sample_times))

                oe_graph_show=[]
                for lam in lam_list:
                    for id_inter_sample_idx in range(len(id_graphon_list)-1):
                        if not args.keep_oe_graphon_size:
                            id_graphon1=random.choice(id_graphon_list)
                            id_graphon2=random.choice(id_graphon_list)
                            min_size=min(id_graphon1.shape[0],id_graphon2.shape[0])//2
                            max_size=max(id_graphon1.shape[0],id_graphon2.shape[0])
                        interoe_graph_new=two_graphon_mixup_random_align(id_graphon1,id_graphon2,min_size=min_size,max_size=max_size,la=lam, num_sample=num_sample)
                        oe_graph_show+=interoe_graph_new
                        new_graph+=interoe_graph_new
                inter_dataset=new_graph
                # dataloader_inter,inter_dataset=prepare_dataloader_x_xs(new_graph,dataset_for_search=dataset_triple[0],args=args)

            if args.search_all and args.mixup_rate:
                dataset_search=mixoe_dataset_all
            else:
                dataset_search= dataset_triple[0]

            multi_dataset=[]
            if args.realoe_rate:
                multi_dataset+=realoe_dataset
            if args.mixup_rate:
                multi_dataset+=prepare_dataset_x_xs_mean(mixup_dataset,dataset_for_search=dataset_search,close_k=args.close_k,args=args)
            if args.inter_rate:
                inter_dataset1=prepare_dataset_x_xs_mean(inter_dataset,dataset_for_search=dataset_search,close_k=args.close_k,args=args)
                multi_dataset+=inter_dataset1
    
            dataloader_oe=DataLoader(multi_dataset, batch_size=args.batch_size, shuffle=True)     
      
        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if trial == 0:
            log.info('================')
            log.info('Exp_type: {}'.format(args.exp_type))
            log.info('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            log.info('num_train: {}'.format(n_train))
            log.info('num_features: {}'.format(dataset_num_features))
            log.info('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            log.info('hidden_dim: {}'.format(args.hidden_dim))
            log.info('num_gc_layers: {}'.format(args.num_layer))
            log.info('num_cluster: {}'.format(args.num_cluster))

            if args.OE==1:
                log.info('====use mix real oe====')
                log.info(f'oe_weight: {args.oe_weight}')

            log.info('================')

        dataloader_len = len(dataloader)
        if args.OE==1:
            dataloader_oe_len = len(dataloader_oe)

            num_repeats = dataloader_len // dataloader_oe_len + 1

            dataloader_oe_upsample = dataloader_oe

        last_oe_thre=0
        for epoch in range(1, num_epoch + 1):
            oe_thre=last_oe_thre
            last_oe_thre=0
            if args.is_adaptive:
                if epoch == 1:
                    weight_b, weight_g, weight_n = 1, 1, 1
                else:
                    weight_g, weight_n = std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_g  + weight_n) / 2
                    weight_g, weight_n = weight_g/weight_sum, weight_n/weight_sum

            model.train()
            loss_all = 0
            if args.is_adaptive:
                loss_b_all, loss_g_all, loss_n_all = [], [], []

            if args.OE==0:
                for data in dataloader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    loss_g = model.calc_loss_g(g_f, g_s)
                    # loss_b = model.calc_loss_b(b, data.idx, cluster_result)
                    loss_n = model.calc_loss_n(n_f, n_s, data.batch)
                    if args.is_adaptive:
                        # loss = weight_b * loss_b.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                        loss = weight_g * loss_g.mean() + weight_n * loss_n.mean()
                        # loss_b_all = loss_b_all + loss_b.detach().cpu().tolist()
                        loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                        loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                    else:
                        # loss = loss_b.mean() + loss_g.mean() + loss_n.mean()
                        loss = loss_g.mean() + loss_n.mean()
                    loss_all += loss.item() * data.num_graphs
                    # log.info('loss_all',loss_all)
                    loss.backward()
                    optimizer.step()

            elif args.OE==1: 
                for data,oe_data in zip(dataloader,dataloader_oe_upsample):
                    data = data.to(device)
                    oe_data=oe_data.to(device)
                    optimizer.zero_grad()
            
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    loss_g = model.calc_loss_g(g_f, g_s)
                    loss_n = model.calc_loss_n(n_f, n_s, data.batch)
                    if args.is_adaptive:
                        loss = weight_g * loss_g.mean() + weight_n * loss_n.mean()
                        loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                        loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                    else:
                        loss = loss_g.mean() + loss_n.mean()
            
                    b_oe, g_f_oe, g_s_oe, n_f_oe, n_s_oe = model(oe_data.x, oe_data.x_s, oe_data.edge_index, oe_data.batch, oe_data.num_graphs)

                    loss_g_oe = model.calc_loss_g(g_f_oe, g_s_oe)

                    loss_n_oe = model.calc_loss_n(n_f_oe, n_s_oe, oe_data.batch)

                    if args.oe_loss_type in ['origin','sigmoid','log','pairlog']:
                        if args.is_adaptive:
                            loss_oe =weight_g * loss_g_oe.mean() + weight_n * loss_n_oe.mean()
                        else:
                            loss_oe = loss_g_oe.mean() + loss_n_oe.mean()
                    

                    if args.oe_loss_type=='baloss':
                        loss_g_oe_ba=baloss(loss_g_oe,gamma=args.gamma)
                        loss_n_oe_ba=baloss(loss_n_oe,gamma=args.gamma)                                           
                    else:
                        raise NotImplementedError 

                    if args.oe_loss_type=='baloss':
                        if args.is_adaptive:
                            loss_oe =weight_g *  loss_g_oe_ba+ weight_n * loss_n_oe_ba
                        else:
                            loss_oe =loss_g_oe_ba + loss_n_oe_ba
                    loss_total=loss+loss_oe*args.oe_weight

                    loss_all += (loss_total.item()) * data.num_graphs
                    # log.info('loss_all',loss_all)
                    loss_total.backward()
                    optimizer.step()
            else:
                raise NotImplementedError

            log.info('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:

                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)

            if epoch % args.eval_freq == 0:

                model.eval()
                y_score_all = []
                y_true_all = []

                test_data_list=dataloader_test

                for data in test_data_list:
                    data = data.to(device)
                    b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                    y_score_g = model.calc_loss_g(g_f, g_s)
                    y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
                    if args.is_adaptive:
                        y_score = (y_score_g - mean_g)/std_g + (y_score_n - mean_n)/std_n
                    else:
                        y_score = y_score_g + y_score_n
                    y_true = data.y 
                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()
    
                auc = skm.roc_auc_score(y_true_all, y_score_all)

                log.info('[EVAL] Epoch: {:01d} | AUC:{:.4f}'.format(epoch, auc))

        log.info('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(trial, auc))
        aucs.append(auc)

    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # log.info the information to the console

    log.info(args)
    log.info('[FINAL RESULT] AVG_AUC: {:.2f}+-{:.2f}'.format(avg_auc * 100, std_auc * 100))
