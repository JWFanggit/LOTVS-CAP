import torch
import torch.nn as nn
import torch.nn.functional as F
class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(512,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a
class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b,s,n, _ = x.shape
        q = self.query(x, dims=([3], [0]))
        k = self.key(x, dims=([3], [0]))
        v = self.value(x, dims=([3], [0]))
        q = q.permute(0, 1, 3, 2,4)
        k = k.permute(0, 1, 3, 2,4)
        v = v.permute(0, 1, 3, 2,4)
        print(q.shape,'qq')
        print(k.shape,'kk')
        print(v.shape,'vv')

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 1, 3, 2, 4)
        print(out.shape)

        out = self.out(out, dims=([3, 4], [1, 2]))

        return out

x=torch.randn(2,2,1,512)
net=SelfAttention( in_dim=512, heads=8, dropout_rate=0.1)
out=net(x)
print(out.shape)
