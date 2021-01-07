'''
Author: sigmoid
Description: 修改模型实现方式，加入Pos
Email: 595495856@qq.com
Date: 2020-12-18 13:04:36
LastEditTime: 2021-01-07 11:07:03
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from modules.convLSTM import ConvLSTM, ConvSigmoid
from modules.attention import PositionAttention, CoverageAttention
n = 256
n_prime = 512
decoder_conv_filters = 256
gru_hidden_size = 256
embedding_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
class Maxout(nn.Module):
    """
    Maxout makes pools from the last dimension and keeps only the maximum value from
    each pool.
    """

    def __init__(self, pool_size):
        """
        Args:
            pool_size (int): Number of elements per pool
        """
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        [*shape, last] = x.size()
        out = x.view(*shape, last // self.pool_size, self.pool_size)
        out, _ = out.max(-1)
        return out

class PositionEnhance(nn.Module):
    """  
        PositionAware+PosAttention
    """
    def __init__(
        self,
        position_dim=512,
        output_size=64,
        device=device
        ):
        super(PositionEnhance, self).__init__()
        self.context_size = 684
        self.device = device
        # self.conv1 = nn.Conv2d(in_channels=self.context_size,
        #                       out_channels=output_size,
        #                       kernel_size=3,
        #                       padding=1,
        #                       bias=True)
        # self.conv2 = nn.Conv2d(in_channels=output_size,
        #                       out_channels=output_size,
        #                       kernel_size=3,
        #                       padding=1,
        #                       bias=True)                    
        # self.convlstm = ConvLSTM(self.context_size, 64, (3, 3), 2, True, True, False)
        self.convsigmoid = ConvSigmoid(self.context_size, 64, (3, 3), True, True, False)
        self.positin_attn = PositionAttention(
            self.context_size,
            decoder_conv_filters,
            kernel_size=(11, 11),
            padding=5,
            device=device
        )

    def forward(self, feature, query):
        # feature:(bs, c, h, w)
        bs = feature.size(0)
        # _, last_state_list = self.convlstm(feature.unsqueeze(1))
        # key = self.conv2(torch.sigmoid(self.conv1(feature)))
        # _, last_state_list = self.convgru(feature.unsqueeze(1))
        key = self.convsigmoid(feature)
        gt_hat = self.positin_attn(feature, query, key) # gt_hat:(bs, context_size, L)
        return gt_hat

class DynamicallyFusing(nn.Module):
    def __init__(
        self, 
        context_size=684,
        output_size=256):
        super(DynamicallyFusing, self).__init__()
        self.fc_Wa = nn.Linear(context_size*2, output_size, bias=False)
        self.fc_Wp = nn.Linear(context_size*2, output_size, bias=False)
        # init
        nn.init.xavier_normal_(self.fc_Wa.weight)
        nn.init.xavier_normal_(self.fc_Wp.weight)
        
    def forward(self, gt, gt_hat):
        gt_cat = torch.cat((gt, gt_hat), 1).transpose(1, 2)
        wt = F.sigmoid(self.fc_Wa(gt_cat)) # (bs, L, c)
        gtf = torch.mul(wt, self.fc_Wp(gt_cat))
        return gtf.transpose(1, 2)

class Decoder(nn.Module):
    """Decoder
    GRU based Decoder which attends to the low- and high-resolution annotations to
    create a LaTeX string.
    """
    def __init__(
        self,
        num_classes,
        batch_size,
        input_size=256,
        hidden_size=256,
        embedding_dim=256,
        pos_embedding_dim=512,
        checkpoint=None,
        device=device,
    ):
        """
        Args:
            num_classes (int): Number of symbol classes
            low_res_shape ((int, int, int)): Shape of the low resolution annotations
                i.e. (C, W, H)
            high_res_shape ((int, int, int)): Shape of the high resolution annotations
                i.e. (C_prime, 2W, 2H)
            input_size: dimension is same with dynamical_fus output dim
            hidden_size (int, optional): Hidden size of the GRU [Default: 256]
            embedding_dim (int, optional): Dimension of the embedding [Default: 256]
            Pos_embedding (int, optional): Dimension of the embedding [Default: 512]
            checkpoint (dict, optional): State dictionary to be loaded
            device (torch.device, optional): Device for the tensors
        """
        super(Decoder, self).__init__()

        context_size = 684 
        self.bs = batch_size
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.pos_embedding = nn.Embedding(cfg.maxlen, pos_embedding_dim)
        self.gru1 = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.coverage_attn = CoverageAttention(
            context_size,
            decoder_conv_filters,
            kernel_size=(11, 11),
            padding=5,
            device=device,
        )
        self.pos_enhance = PositionEnhance()
        self.dynamical_fus = DynamicallyFusing()
        
        self.W_o = nn.Parameter(torch.empty((num_classes, embedding_dim // 2)))
        self.W_s = nn.Parameter(torch.empty((embedding_dim, hidden_size)))
        self.W_c = nn.Parameter(torch.empty((embedding_dim, input_size)))
    
        self.maxout = Maxout(2)
        self.hidden_size = hidden_size
        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.W_s)
        nn.init.xavier_normal_(self.W_c)

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self, batch_size, feat_shape):
        self.coverage_attn.reset_alpha(batch_size, feat_shape[2]*feat_shape[3])
    
    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since only the previous symbol is provided, not a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, x, hidden, feature, idx):
        embedded = self.embedding(x)
        pos = torch.LongTensor([idx]*self.bs).view(-1, 1).cuda()  # pos
        pos_embedded = self.pos_embedding(pos)
        
        pred, _ = self.gru1(embedded, hidden)
        gt = self.coverage_attn(feature, pred) # (bs, 684, L) L=h*w
        
        gt_hat = self.pos_enhance(feature, pos_embedded)

        context = self.dynamical_fus(gt, gt_hat) # (bs, c, L)
        context = context.sum(2)
        new_hidden, _ = self.gru2(context.unsqueeze(1), pred.transpose(0, 1))
        
        w_s = torch.matmul(self.W_s, new_hidden.squeeze(1).t()).t()
        w_c = torch.matmul(self.W_c, context.t()).t()
        out = embedded.squeeze(1) + w_s + w_c
        out = self.maxout(out)
        out = torch.matmul(self.W_o, out.t()).t()
        return out, new_hidden.transpose(0, 1)  
