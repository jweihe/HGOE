import torch
import torch.nn.functional as F

def entropy_loss(x):
    # Normalize x to get a distribution
    p = F.softmax(x, dim=0)
    # Compute the entropy of the distribution
    H = -torch.sum(p * torch.log(p + 1e-10))  # adding a small value to avoid log(0)
    
    return H


def baloss(x, gamma=2.0):
    p = torch.sigmoid(x)
    FL = (2 - p) ** gamma * torch.log(p + 1e-10)
    return torch.mean(FL)


