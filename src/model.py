# coding=UTF-8
import os
import math
import sys
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import cv2

os.environ['CUDA_VISIBLE_DEVICES']= '0'
device = ("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

np.set_printoptions(suppress=True)
from src.bert import FuseModel,TextModel,opt
from src.Gaussian import get_gaussian_kernel
from transformers import BertTokenizer
from src.Transformer import SelfAttention
from src.A import LAL
import torch


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""
    r“”“应用图卷积的基本模块。
    arg：
        in_channels（int）：输入序列数据中的通道数
        out_channels（int）：卷积产生的通道数
        kernel_size（int）：图卷积内核的大小
        t_kernel_size（int）：时间卷积核的大小
        t_stride（整数，可选）：时间卷积的跨度。默认值：1
        t_padding（int，可选）：在控件的两边都添加了时间零填充
            输入。默认值：0
        t_dilation（整数，可选）：时间内核元素之间的间距。
            默认值：1
        偏见（布尔型，可选）：如果为``True''，则向输出添加可学习的偏见。
            默认值：``True``
    形状：
        -Input [0]：以（n,in_channels,T_ {in},V）格式输入图形序列
        -Input [1]：以（K,V,V）格式输入图邻接矩阵
        -output[0]：Outpu图形序列,格式为（N,out_channels,T_ {out},V）`
        -Output [1]：输出数据的图形邻接矩阵，格式为（K,V,V）`
        哪里
            ：ma：`N`是批处理大小，
            ：math：`K`是空间内核大小，如：math：`K == kernel_size [1]`，
            ：math：`T_ {in} / T_ {out}`是输入/输出序列的长度,
            V是图形节点的数量。
    “”

    The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,  # 768
                 out_channels,  # 2
                 kernel_size,  # [1,2]
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        # dilation 3*3卷积间隔一个虽然还是3*3但效果类似5*5，维度变化是按5*5
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,  # 5个通道5个卷积核
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
    def forward(self, x, A):
        A=A.to(device)
        x = self.conv(x)
        x=torch.squeeze(x,2)
        # x = torch.einsum('nctv,tvw->nctw',
        x = torch.einsum('ncv,nvw->ncw',
                         (x, A))
        return x.contiguous(), A


class GRUNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=2, n_layers=1, dropout=[0,0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        # self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.maxpool=nn.AdaptiveMaxPool1d(1)


    def forward(self, x, h):
        # print(x.shape)
        # print(h.shape)
        x=x.permute(0,2,1).contiguous()
        # x=self.avgpool(x)
        x=self.maxpool(x)
        out, h = self.gru(x.permute(0,2,1).contiguous(), h)
        # print(out.shape)
        out = F.dropout(out[:,-1],self.dropout[0])
        # print(out.shape)
        out = self.relu(self.dense1(out))
        out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        return out

class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        return out





class PositionEmbs(nn.Module):
    def __init__(self, num_patches, c_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(2, c_dim,1,num_patches+1))
        # print(self.pos_embedding.shape)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,  # 2
                 out_channels,  # 5
                 kernel_size,  # [3,8]
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  #
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.MLP = LAL()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,  # 2
                    out_channels,  # 5
                    kernel_size=1,
                    stride=(stride, 1)),  # [1,1]
                nn.BatchNorm2d(out_channels),
            )  # [8,57]————>  n=8-1+1=8 [8,57]

        self.prelu = nn.PReLU()

    def forward(self, x):
        A = self.MLP(x)
        x = torch.unsqueeze(x, 2)
        x = x.permute(0, 3, 2, 1).contiguous()
        res = self.residual(x)
        x, A = self.gcn(x, A)
        # x=torch.unsqueeze(x,2)
        # # x = self.tcn(x) + res  # tcn[128,5,8,57]相加还是
        # x=x+res
        # x=torch.squeeze(x,2)
        # if not self.use_mdn:
        #     x = self.prelu(x)  # 激活层
        return x


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - torch.min(s_map))/((torch.max(s_map)-torch.min(s_map))*1.0)
    # a-mina/(maxa-mina)*1
    return norm_s_map



class conv_deconv(nn.Module):
    def __init__(self):
        super(conv_deconv,self).__init__()
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=512,out_channels=64,kernel_size=3)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.swish4=nn.ReLU()
        self.upsampling=nn.Upsample(size=(58,58),mode='bilinear', align_corners=True)
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=64,out_channels=16,kernel_size=3)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.swish5=nn.ReLU()
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=5)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.swish6=nn.ReLU()
        self.BN=nn.BatchNorm2d(64)
        self.Attention= SelfAttention(512,8 ,0.1)
        self.gussian = get_gaussian_kernel(3, 1.5, 1)
    def forward(self,x):
        x= self.Attention(x)
        B, new_HW, C = x.shape
        x = x.view(B, C, 8, 8)
        out=self.deconv1(x)
        out=self.BN(out)
        out=self.swish4(out)
        out=self.upsampling(out)
        out=self.deconv2(out)
        out=self.swish5(out)
        # out=self.upsampling2(out)
        out=self.deconv3(out)
        out=self.swish6(out)
        out = self.gussian(out)
        out = normalize_map(out)
        return(out)
# class Decoder2D(nn.Module):
#     def __init__(self, in_channels=512, out_channels=1,
#                  features = [64, 16, 1]):
#         super().__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.decoder_1 = nn.Sequential(
#             nn.Conv2d(in_channels, features[0], 3, padding=1),
#             nn.BatchNorm2d(features[0]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_2 = nn.Sequential(
#             nn.Conv2d(features[0], features[1], 3, padding=1),
#             nn.BatchNorm2d(features[1]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_3 = nn.Sequential(
#             nn.Conv2d(features[1], features[2], 3, padding=1),
#             nn.BatchNorm2d(features[2]),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.Attention= SelfAttention(512,8 ,0.1)
#         # self.decoder_4 = nn.Sequential(
#         #     nn.Conv2d(features[2], features[3], 3, padding=1),
#         #     nn.BatchNorm2d(features[3]),
#         #     nn.ReLU(),
#         #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         # )
#         # self.decoder_5 = nn.Sequential(
#         #     nn.Conv2d(features[3], features[4], 3, padding=1),
#         #     nn.BatchNorm2d(features[4]),
#         #     nn.ReLU(),
#         #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         # )
#         # self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)
#
#
#
#     def forward(self,x):
#         # x = x.permute(0, 2, 1)
#         x= self.Attention(x)
#         B, new_HW, C = x.shape
#         x = x.view(B, C, 8, 8)
#         x = self.decoder_1(x)
#         x = self.decoder_2(x)
#         x = self.decoder_3(x)
#         # x = self.decoder_4(x)
#         # x = self.decoder_5(x)
#         # x = self.final_out(x)
#         # x = self.sigmoid(x)
#         # x = self.gussian(x)
#         x = normalize_map(x)
#
#         return x
def padding(squence, max_length, pad_token=0):
    # squence is the text that needs to be processed, pad_token is the token of the padding, and defaults to 0
    # If padding_length is greater than 0, it indicates the number of pad_tokens to be added. If it is smaller than 0,
    # it indicates the number of tokens to be truncated
    padding_length = max_length - len(squence)
    return squence + [pad_token] * padding_length
class Text(nn.Module):
    def __init__(self):
        super(Text,self).__init__()
        self.textNet = TextModel(opt)
    def forward(self, x):
        tokenizer = BertTokenizer.from_pretrained(r'/bert/modeling_bert/bert-base-uncased-vocab.txt')
        max_len = 15
        res=[]
        for text in x:
            inputs = tokenizer.encode_plus(
                text,  # input
                add_special_tokens=True,
                max_length=max_len,  # max length of words
                truncation=True
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            # initialized attention_mask or nput_mask
            attention_mask = [1] * len(input_ids)
            input_ids = padding(input_ids, max_len, pad_token=0)
            attention_mask = padding(attention_mask, max_len, pad_token=0)
            token_type_ids = padding(token_type_ids, max_len, pad_token=0)
            res.append((input_ids, attention_mask, token_type_ids))
        all_input_ids = torch.tensor([x[0] for x in res], dtype=torch.int64).to(device)
        all_attention_mask = torch.tensor([x[1] for x in res], dtype=torch.int64).to(device)
        # all_token_type_ids = torch.tensor([x[2] for x in res], dtype=torch.int64)
        return all_input_ids,all_attention_mask



class accident(nn.Module):
    def __init__(self,h_dim,n_layers,depth,adim,heads,num_tokens,c_dim,s_dim1,s_dim2,keral,num_class):
        super(accident, self).__init__()
        self.depth=depth
        self.n_layers=n_layers
        self.h_dim=h_dim
        self.adim=adim
        self.heads=heads
        self.num_pathches=num_tokens
        self.c_dim=c_dim
        self.sdim1=s_dim1
        self.sdim2=s_dim2
        self.keral=keral
        self.num_class=num_class
        self.text=Text()
        self.fusion=FuseModel(opt)
        self.gru_net=GRUNet(h_dim+h_dim, h_dim,self.num_class,n_layers, dropout=[0.3,0.2])
        self.features=st_gcn(self.sdim1 ,self.sdim2 ,self.keral)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.deconv=conv_deconv()

    def forward(self,x,z,y,toa,w):
        #x:rgb、z:foucs、y:label(positive,negative)、toa:time to accident、w:word(text)
        losses = {'total_loss': 0}
        all_output=[]
        x_11 = x
        # hh is the initial hidden state
        hh = Variable(torch.zeros(self.n_layers, x_11.size(0), self.h_dim))
        hh=hh.to(device)
        for i in range(x.size(1)):
            x1 =x_11[:, i]
            x2 =z[:, i]
            tokens_tensor, input_masks_tensors=self.text(w)
            x= self.fusion(tokens_tensor,input_masks_tensors,x1)
            x = self.features(x)
            x=x.permute(0,2,1).contiguous()
            output1 = self.gru_net(x, hh)
            #also can output foucs_p
            foucs_p=self.deconv(x)
            L1 = self._exp_loss(output1, y,i, toa, fps=30.0)
            L2=self.kl_loss(x2,foucs_p )
            loss_sum=(5*L1+L2).mean()
            losses['total_loss'] += loss_sum
            all_output.append(output1)
        return losses,all_output



    def _exp_loss(self, pred, target, time, toa, fps=30.0):
            target_cls = target[:, 1]
            target_cls = target_cls.to(torch.long)
            penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype),
                                 (toa.to(pred.dtype) - time - 1) / fps)
            pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
            neg_loss = self.ce_loss(pred, target_cls)
            loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
            return loss

    def kl_loss(self,y_true, y_pred, eps=1e-07):
        P = y_pred
        P = P / (eps + torch.sum(P,dim=(0,1,2,3),keepdim=True ))
        Q = y_true
        Q = Q / (eps + torch.sum(Q,dim=(0,1,2,3),keepdim=True))
        kld = torch.sum(Q * torch.log(eps + Q / (eps + P)), dim=(0,1,2,3))
        return kld





