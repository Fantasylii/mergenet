import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from .hyper_module import HyperNetwork
from torch.nn.utils.parametrizations import spectral_norm

class Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_q, d_k, d_v, d_attention, h, dropout=0.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(Attention, self).__init__()
        self.fc_q = nn.Linear(d_q, h * d_attention)
        self.fc_k = nn.Linear(d_k, h * d_attention)
        self.fc_v = nn.Linear(d_v, h * d_attention)
        self.fc_output = nn.Linear(h * d_attention, d_q)
        self.dropout=nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_q, eps=1e-12)

        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_attention = d_attention
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        nq = queries.shape[0]
        nk = keys.shape[0]
        q = self.fc_q(queries).view(nq, self.h, self.d_attention).permute(1, 0, 2)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(nk, self.h, self.d_attention).permute(1, 2, 0)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(nk, self.h, self.d_attention).permute(1, 0, 2)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_attention)  # (b_s, h, nq, nk)
        print(att.size())
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(1, 0, 2).contiguous().view(nq, self.h * self.d_attention)  # (b_s, nq, h*d_v)
        # print(out.shape)
        out = queries + self.fc_output(out)  # (b_s, nq, d_model)
        out = self.norm(out)
        return out
    
class ParamOutput(nn.Module):
    def __init__(self, dimension):
        super(ParamOutput, self).__init__()
        self.dense_up = spectral_norm(nn.Linear(dimension, 2 * dimension))
        self.dense_dowm = spectral_norm(nn.Linear(2 * dimension, dimension))
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(dimension, eps=1e-12)

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.gelu(self.dense_up(x))
        out = self.layernorm(self.dropout(self.dense_dowm(out)) + x)
        return out
    
class ParamOutput2(nn.Module):
    def __init__(self, dimension):
        super(ParamOutput2, self).__init__()
        self.dense = nn.Linear(dimension, dimension)
        nn.init.kaiming_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.0)
        self.dropout = nn.Dropout(0.3)
        self.layernorm = nn.LayerNorm(dimension, eps=1e-12)

    def forward(self, x):
        out = self.dropout(self.dense(x))
        out = self.layernorm(out + x)
        return out

class ParamAttentionLayer(nn.Module):
    def __init__(self, a_size, b_size, d_attention, h, mode):
        super(ParamAttentionLayer, self).__init__()
        self.a_size = a_size
        self.b_size = b_size
        self.d_attention = d_attention
        self.mode = mode
        d_attention_x_x = (max(int((self.a_size[1] + self.b_size[1]) // 2), 128)) // h
        d_attention_x_y = (max(int((self.a_size[1] + self.b_size[0]) // 2), 128)) // h
        d_attention_y_x = (max(int((self.a_size[0] + self.b_size[1]) // 2), 128)) // h 
        d_attention_y_y = (max(int((self.a_size[0] + self.b_size[0]) // 2), 128)) // h
        self.attention_x_x = Attention(self.a_size[1], self.b_size[1], self.b_size[1], d_attention_x_x, h)
        self.attention_x_y = Attention(self.a_size[1], self.b_size[0], self.b_size[0], d_attention_x_y, h)
        self.attention_y_x = Attention(self.a_size[0], self.b_size[1], self.b_size[1], d_attention_y_x, h)
        self.attention_y_y = Attention(self.a_size[0], self.b_size[0], self.b_size[0], d_attention_y_y, h)
        self.output = ParamOutput(self.a_size[1])
        # self.weights = nn.Parameter(torch.ones(4, requires_grad=True))

    def forward(self, param_a, param_b):
        # if self.mode == 'b':
        #     out_x_x = self.attention_x_x(param_a, param_b, param_b)
        #     out = out_x_x 
        #     return out
        size_a = param_a.shape
        # weights = F.softmax(self.weights, dim=-1)
        out_x_x = self.attention_x_x(param_a, param_b, param_b)
        out_x_y = self.attention_x_y(param_a, param_b.t(), param_b.t())
        out_y_x = self.attention_y_x(param_a.t(), param_b, param_b)
        out_y_y = self.attention_y_y(param_a.t(), param_b.t(), param_b.t())
        # out = out_x_x * weights[0] + out_x_y * weights[1] + out_y_x * weights[2] + out_y_y * weights[3]
        out = (out_x_x  + out_x_y + out_y_x.t() + out_y_y.t()) / 4
        out = self.output(out)
        return out
    
class ParamAttention(nn.Module):
    def __init__(self, config, mode='a'):
        super(ParamAttention, self).__init__()
        self.num_layers = config[f'num_layers']
        size1 = config['d_attention'] if mode == 'a' else config['d_attention']
        size2 = config['d_attention'] if mode == 'a' else config['d_attention']
        layer = [ParamAttentionLayer(size1, size2, config['d_attention'], config['h'], mode=config['mode']) for i in range(self.num_layers)]
        self.layer = nn.ModuleList(layer)
        self.linear1 = nn.Linear(size1[1], size1[1])
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)

    
    def forward(self, param_a, param_b):
        
        a_conv = param_a['conv'].squeeze()
        # a_conv_bias = param_a['conv_bias']
        # print(a_conv.size(), param_a['conv_bias'].size())
        attention_a = a_conv

        for i in range(self.num_layers):
            attention_a = self.layer[i](attention_a, param_b['linear_weight'])
        # attention_a = self.linear1(attention_a)
        
        return attention_a.reshape(param_a['conv'].size())

if __name__ == '__main__':
    config = dict()
    config['a_size'] = [100, 512]
    config['b_size'] = [200, 1024]
    config['d_attention'] = 64
    config['h'] = 8
    config['mode'] = 'a'
    config['num_layers'] = 3
    a = torch.randn(100, 512)
    b = torch.randn(200, 1024)
    sa = ParamAttention(config)
    out_a, out_b=sa(a,b)
    print('111111111', out_a.shape, out_b.shape)
        