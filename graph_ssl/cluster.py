import faiss
import numpy as np
import torch

def run_ssl_kmeans(x, args):

    results = {}

    d = x.shape[1]
    k = args.id_cluster_num
    
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

def get_ssl_cluster_result(args,model,dataloader,n_train,device='cuda'):
    model.encoder_model.eval()
    embeddding_all = torch.zeros((n_train, model.embedding_dim*2))
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            embeddding=model.encoder_fea(data)
            # print('data.idx',data.idx)
            embeddding_all[data.idx] =embeddding.detach().cpu()
    cluster_result = run_ssl_kmeans(embeddding_all.numpy(), args)
    return cluster_result

def split_dataset_by_ssl_feature(args,model,dataset,dataloader,n_train,device='cuda'):
    cluster_result = get_ssl_cluster_result(args,model,dataloader,n_train,device)
    im2cluster = cluster_result['im2cluster']
    centroids = cluster_result['centroids']
    density = cluster_result['density']


    dataset_list=[]
    for i in range(args.id_cluster_num):
        dataset_list.append([])
    for i in range(n_train): 
        dataset_list[im2cluster[i]].append(dataset[i])
    return dataset_list,im2cluster,centroids,density

