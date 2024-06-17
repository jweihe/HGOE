import torch
import os.path as osp


import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator,LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset



import warnings
warnings.filterwarnings("ignore")

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss



def test_ogb_graph(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(torch.round(data.y.float()).int())
        # y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    print('y',y[0:10])
    # print('y',y)
    emb=x.detach().cpu()
    acc_val, acc = evaluate_embedding(emb, y.detach().cpu())
    return acc_val, acc

def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        # print('data.y',data.y)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)


    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True,max_iter=3000)(x, y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')

    DS='ogbg-molbbbp'

    TU = not DS.startswith('ogbg-mol')

    if TU:
        dataset = TUDataset(path, name=DS)
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)

    dataloader = DataLoader(dataset, batch_size=128) 
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    epochs=100
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1,epochs+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    # data = dataset[0].to(device)
    # test_result = test(encoder_model, dataloader)
    # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    if not TU:
        acc,acc_val = test_ogb_graph(encoder_model, dataloader)
        print(f'(E): test acc={acc:.4f}, acc_val={acc_val:.4f}')
        # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    else:
        test_result = test(encoder_model, dataloader)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

class GraphCLModel:
    def __init__(self, dataloader, num_feature,embedding_dim=32,device='cuda',pn=0.1,pf=0.1,pe=0.1,lr=0.1,TU=False):
        # self.dataloader = DataLoader(dataset, batch_size=128) 
        self.dataloader=dataloader
        self.input_dim = num_feature
        self.device = device
        self.TU = TU
        self.embedding_dim=embedding_dim
        self._initialize_model(pn,pf,pe,lr)

    def _initialize_model(self,pn,pf,pe,lr):
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                               A.NodeDropping(pn=pn),
                               A.FeatureMasking(pf=pf), 
                               A.EdgeRemoving(pe=pe)], 1)
        gconv = GConv(input_dim=self.input_dim, hidden_dim=self.embedding_dim, num_layers=2).to(self.device)
        self.encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(self.device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(self.device)

        self.optimizer = Adam(self.encoder_model.parameters(), lr=lr)

    def train_model(self, epochs=100):
        with tqdm(total=epochs, desc='(T)') as pbar:
            for epoch in range(1, epochs+1):
                loss = train(self.encoder_model, self.contrast_model, self.dataloader, self.optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
    
    def encoder_fea(self, data):
        self.encoder_model.eval()
        _, g, _, _, _, _ = self.encoder_model(data.x, data.edge_index, data.batch)
        return g
        
    def test_model(self):
        if not self.TU:
            acc, acc_val = test_ogb_graph(self.encoder_model, self.dataloader)
            print(f'(E): test acc={acc:.4f}, acc_val={acc_val:.4f}')
        else:
            test_result = test(self.encoder_model, self.dataloader)
            print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')



if __name__ == '__main__':
    main()
