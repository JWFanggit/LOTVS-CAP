# -*- coding: UTF-8 -*-
import torch
from torch import nn


#
def calculate_laplacian_with_self_loop(matrix):
     matrix = matrix+torch.eye(matrix.size(0)).cuda()
     # matrix=matrix+b
     row_sum = matrix.sum(1)
     d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
     d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
     normalized_laplacian = (
     matrix.matmul(d_mat_inv_sqrt).transpose(1, 2).matmul(d_mat_inv_sqrt)
     )
     return normalized_laplacian

def data_normal(orign_data):
    d_min=torch.min(orign_data)
    if d_min<0:
        orign_data +=torch.abs(d_min )
        d_min=torch.min(orign_data)
    d_max=torch.max(orign_data)
    dst=d_max-d_min
    normal_data=(orign_data-d_min ).true_divide(dst)
    return normal_data

class LAL(nn.Module):
    def __init__(self):
        super( LAL, self).__init__()

    def forward(self, x):
        out=torch.cdist(x,x,p=2)
        out=data_normal(out)
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                            if i == j:
                                out[:, i, j] = 1
        # out=out.cpu().detach().numpy()
        # for s in range(out.shape[0]):
        #     G=nx.from_numpy_matrix(out[s,:,:])
        #     out[s,:,:]=nx.normalized_laplacian_matrix(G).toarray()
        # out=torch.from_numpy(out).type(torch.float)
        # out=data_normal(out)
        return out







