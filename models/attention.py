'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2020-12-28 12:08:41
LastEditTime: 2020-12-28 13:05:03
'''
import torch
import torch.nn as nn

n = 256
n_prime = 512
decoder_conv_filters = 256
gru_hidden_size = 256
embedding_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoverageAttention(nn.Module):
    """Coverage attention
    The coverage attention is a multi-layer perceptron, which takes encoded annotations
    and creates a context vector.
    """
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        padding=0,
        device=device,
    ):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the coverage
            attn_size (int): Length of the annotation vector
            kernel_size (int): Kernel size of the 1D convolutional layer
            padding (int, optional): Padding of the 1D convolutional layer [Default: 0]
            device (torch.device, optional): Device for the tensors
        """
        super(CoverageAttention, self).__init__()
        
        self.alpha = None
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.conv_Q = nn.Conv2d(1, output_size, kernel_size=kernel_size, padding=padding) # same
        self.fc_Wa = nn.Linear(n, n_prime)
        self.conv_Ua = nn.Conv2d(input_size, n_prime, kernel_size=1)
        self.fc_Uf = nn.Linear(output_size, n_prime)
        self.fc_Va = nn.Linear(n_prime, 1)
        
        # init
        nn.init.xavier_normal_(self.fc_Wa.weight)
        nn.init.xavier_normal_(self.fc_Va.weight)
        nn.init.xavier_normal_(self.fc_Uf.weight)

    def reset_alpha(self, bs, attn_size): 
        """
        Args:
            attn_size: H*W (feature map size)
        """
        self.alpha = torch.zeros((bs, 1, attn_size), device=self.device)

    def forward(self, x, st_hat): 
        bs = x.size(0) # x (bs, c, h, w)

        if self.alpha is None:
            self.reset_alpha(bs, x.size(2)*x.size(3))
            
        beta = self.alpha.sum(1)
        beta = beta.view(bs, x.size(2), x.size(3)) # 当前时间步之前的alpha累加 (bs, attn_size)

        F = self.conv_Q(beta.unsqueeze(1)) # (bs, output_size, h, w)
        F = F.permute(2, 3, 0, 1) # (h, w, bs, output_size)
        cover = self.fc_Uf(F) # (h, w, bs, n_prime)
        key = self.conv_Ua(x).permute(2, 3, 0, 1) # (h, w, bs, n_prime)
        query = self.fc_Wa(st_hat).squeeze(1) #(bs, n_prime)
        
        
        attention_score = torch.tanh(key + query[None, None, :, :] + cover)

        e_t = self.fc_Va(attention_score).squeeze(3) # (h, w, bs)
        e_t = e_t.permute(2, 0, 1).view(bs, -1) # (bs, h*w)
        e_t_exp = torch.exp(e_t)
        e_t_sum = e_t_exp.sum(1)
        alpha_t = torch.zeros((bs, x.size(2)*x.size(3)), device=device) # (bs, attn_size)
        for i in range(bs):
            e_t_div = e_t_exp[i]/(e_t_sum[i]+1e-8)
            alpha_t[i] = e_t_div
        self.alpha = torch.cat((self.alpha, alpha_t.unsqueeze(1)), dim=1)
        gt = alpha_t * x.view(bs, x.size(1), -1).transpose(0, 1) # x(bs, c, attn_size)->x(c, bs, attn_size) gt(c, bs, attn_size)
        # (bs, c, attn_size)
        return gt.transpose(0, 1)

class PositionAttention(nn.Module):
    """Coverage attention
    The coverage attention is a multi-layer perceptron, which takes encoded annotations
    and creates a context vector.
    """
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        padding=0,
        device=device,
    ):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the coverage
            attn_size (int): Length of the annotation vector
            kernel_size (int): Kernel size of the 1D convolutional layer
            padding (int, optional): Padding of the 1D convolutional layer [Default: 0]
            device (torch.device, optional): Device for the tensors
        """
        super(PositionAttention, self).__init__()
        self.device = device

        self.fc_Wa = nn.Linear(n, n_prime)
        self.conv_Ua = nn.Conv2d(64, n_prime, kernel_size=1)
        self.fc_Va = nn.Linear(n_prime, 1)
        
        nn.init.xavier_normal_(self.fc_Wa.weight)
        nn.init.xavier_normal_(self.fc_Va.weight)

    def forward(self, feature, query, key): 
        # key:(bs, c, h, w), feature:(bs, c, h, w)
        bs = feature.size(0) 
        
        key = self.conv_Ua(key).permute(2, 3, 0, 1) # (h, w, bs, n_prime)
        
        attention_score = torch.tanh(key + query.squeeze(1)[None, None, :, :])
        e_t = self.fc_Va(attention_score).squeeze(3)
        e_t = e_t.permute(2, 0, 1).contiguous()
        e_t = e_t.view(bs, -1) # (bs, h*w)
        e_t_exp = torch.exp(e_t)
        e_t_sum = e_t_exp.sum(1)
        alpha_t = torch.zeros((bs, feature.size(2)*feature.size(3)), device=self.device) # (bs, L)
        for i in range(bs):
            e_t_div = e_t_exp[i]/(e_t_sum[i]+1e-8)
            alpha_t[i] = e_t_div
        gt_hat = alpha_t * feature.view(bs, feature.size(1), -1).transpose(0, 1) # x(bs, c, L)->x(c, bs, L) gt_hat:(bs, c, L)
        return gt_hat.transpose(0, 1) 
 