import torch.nn.modules as nn
import torch
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
from src.pre_model import RobertaEncoder
import copy
from transformers import logging
from src.Transformer import SelfAttention
logging.set_verbosity_error()
import argparse
parse = argparse.ArgumentParser()
parse.add_argument('-text_model', type=str, default='bert-base', help='language model')
parse.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')
parse.add_argument('-image_size', type=int, default=224, help='Image dim')
parse.add_argument('-image_output_type', type=str, default='all',
                   help='"all" represents the overall features and regional features of the picture, and "CLS" represents the overall features of the picture')
parse.add_argument('-text_length_dynamic', type=int, default=1, help='1: Dynamic length; 0: fixed length')
parse.add_argument('-fuse_type', type=str, default='att', help='att, ave, max')
parse.add_argument('-tran_dim', type=int, default=120,
                   help='Input dimension of text and picture encoded transformer')
parse.add_argument('-tran_num_layers', type=int, default=3, help='The layer of transformer')
parse.add_argument('-image_num_layers', type=int, default=3, help='The layer of image transformer')
parse.add_argument('-l_dropout', type=float, default=0.1,
                   help='classify linear dropout')

# 布尔类型的参数
parse.add_argument('-fixed_image_model', action='store_true', default=False, help='是否固定图像模型的参数')
parse.add_argument('-adim', type=int, default=512,
                   help='')
parse.add_argument('-heads', type=int, default=8,
                   help='the number heads of attention')
parse.add_argument('-num_tokens', type=int, default=49,
                   help='the number of image tokens')
parse.add_argument('-c_dim', type=int, default=512,
                   help='')
parse.add_argument('-s_dim1', type=int, default=120,
                   help='')
parse.add_argument('-s_dim2', type=int, default=512,
                   help='')
parse.add_argument('-keral', type=list, default=[3,4],
                   help='')
parse.add_argument('-num_class', type=int, default=2,
                   help='')
parse.add_argument('-max_length', type=int, default=15,
                   help='the length of word')
opt = parse.parse_args()





class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class BertClassify(nn.Module):
    def __init__(self, opt, in_feature, dropout_rate=0.1):
        super(BertClassify, self).__init__()
        self.classify_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feature, 3),
            ActivateFun(opt)
        )

    def forward(self, inputs):
        return self.classify_linear(inputs)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        # the root_path of bert model
        abl_path = '.../bert/'

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'modeling_bert/bert-base-uncased-config.json')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'modeling_bert/bert-base-uncased-pytorch_model.bin', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output




class embeding(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size,
                 patch_size,
                 emb_dim):
        self.image_size=image_size
        self.patch_size=patch_size
        self.emb_dim=emb_dim
        h, w = image_size
        super(embeding, self).__init__()
        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        # self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw),stride=(fh,fw))
        self.downsampling=nn.Conv2d(emb_dim,120,kernel_size=(1,1),stride=(2,2))



    def forward(self, x):
        emb = self.embedding(x)
        emb=self.downsampling(emb )
        emb = emb.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = emb.shape
        emb = emb.reshape(b,h * w,c)
        return emb
class F_block(nn.Module):
    def __init__(self):
        super(F_block, self).__init__()
    def forward(self,image_init,text_image_init):
        text_image_init = torch.cat(((torch.mul(image_init, text_image_init[:, 0:opt.num_tokens, :])+text_image_init[:, 0:opt.num_tokens, :]),text_image_init[:, opt.num_tokens:opt.num_tokens+opt.max_length, :]), dim=1)
        return text_image_init

class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.text_model = TextModel(opt)
        self.image_model = embeding(image_size=(224,224),patch_size=(16,16),emb_dim=768)
        self.text_config = copy.deepcopy(self.text_model.get_config())
        # self.image_config = copy.deepcopy(self.text_model.get_config())
        self.text_config.num_attention_heads = opt.tran_dim // 15
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers
        self.num_flayers=2
        self.att=SelfAttention(120)
        # self.PE = torch.nn.Parameter(torch.randn(1,49, 120))
        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        if self.fuse_type == 'att':

            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )
        self.fblock= nn.ModuleList([F_block() for _ in range(self.num_flayers)])

    def forward(self, text_inputs, bert_attention_mask, image_inputs):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        # text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)
        image_init=self.image_model(image_inputs)
        text_init =self.att(text_init)
        image_init = self.att(image_init)
        text_image_cat = torch.cat((image_init,text_init), dim=1)
        text_image_transformer = self.text_image_encoder(text_image_cat,
                                                 attention_mask=None,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=True ,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state

        for k in range(self.num_flayers):
            text_image=self.fblock[k](image_init,text_image_transformer)
            text_image=self.text_image_encoder(text_image,
                                                 attention_mask=None,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=True ,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
            text_image_now= text_image.last_hidden_state
            text_image_last=text_image_transformer
            image_init=text_image_last[:,0:opt.num_tokens,:]
            text_image_transformer=text_image_now
        return text_image_transformer



