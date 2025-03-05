import torch.nn as nn
import torch
device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
 


class EML(nn.Module):
    def __init__(self):
        super(EML, self).__init__()
        
    def forward(self,x,y,h):

        epsilon=x-y
        e = epsilon[torch.randperm(epsilon.shape[0])[0:epsilon.shape[0]//4]]
        dist1=-2*torch.mul(epsilon,e.T)
        dist2=torch.sum(torch.square(epsilon), axis=1, keepdims=True)
        dist3 = torch.sum(torch.square(e), axis=1, keepdims=True).T
        dist = torch.abs(dist1+dist2+dist3)
        h = h.view(-1, 1)
        k_hat_matrix = torch.exp(- dist / torch.pow( h, 2)) / h
        k_hat_matrix[k_hat_matrix < 1e-5] = 1e-5
        loss_matrix = -torch.log(torch.mean(k_hat_matrix, dim=1))
        loss = torch.mean(loss_matrix)

        return loss
    
def Variable_bandwidth(x,bandwidth):
    k = int(x.shape[0]*bandwidth)
    dist1=-2*torch.mul(x,x.T)
    dist2=torch.sum(torch.square(x), axis=1, keepdims=True)
    dist3 = dist2.T
    dist = torch.sqrt(dist1+dist2+dist3)
    var_bw=torch.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        dist_k_min=torch.argsort(dist[i])[:k]
        var_bw[i]=torch.max(x[dist_k_min])-torch.min(x[dist_k_min])
    var_bw=var_bw.to(device=device, dtype=torch.float32)
    return var_bw