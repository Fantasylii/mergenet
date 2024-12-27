import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HyperConv_Module(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        z_in_dim: int,
        z_out_dim: int,
        bias: bool=True,
    ) -> None:
        super(HyperConv_Module, self).__init__()
        #2048, 512-> 2048, 320
        self.linear1 = nn.Linear(z_out_dim, 1024)
        self.linear2 = nn.Linear(1024, 1281)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

        self.in_features = in_features
        self.out_features = out_features
        self.z_in_dim = z_in_dim
        self.z_out_dim = z_out_dim
        self.z = None
#         self.bias = nn.Parameter(torch.zeros(100))
    
    def forward(self, z):
        if z.size()[2] > 1:
            aug = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            z = aug(z)
        size = z.size()
        param = z.reshape(size[0], size[1])
        param = self.relu(self.linear1(param))
        # param = torch.transpose(param, 0, 1)
        param = self.linear2(param)
        # param = torch.transpose(param, 0, 1)
        return param

class HyperNetwork(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(HyperNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z = None
        # print(hidden_size)#########################################
        self.linear1 = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        self.linear2 = nn.Linear(1024, out_dim)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.LayerNorm = nn.LayerNorm(out_dim, eps=1e-6)
        # self.param = nn.Sequential(
        #     linear1,
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     linear2,
        #     # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        # )
      
    def forward(self, z):
        # z = torch.transpose(z, 0, 1)
        y = self.relu(self.linear1(z))
        # z = torch.transpose(z, 0, 1)
        # y = self.linear2(y)
        y = self.LayerNorm(y + z)
        return z

class HyperLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        z_in_dim: int,
        z_out_dim: int,
        bias: bool=True,
    ) -> None:
        super(HyperLinear, self).__init__()
        self.hypernetwork = nn.Sequential(
            HyperNetwork(z_in_dim, z_out_dim),
            # HyperNetwork(z_in_dim, z_out_dim),
            # HyperNetwork(z_in_dim, z_out_dim),
        )
#         print(in_features, out_features, z_in_dim,z_out_dim)
        self.z = [None, None, None]
        # linear1 = nn.Linear(768, 1024)
        # nn.init.kaiming_normal_(linear1.weight)
        # nn.init.constant_(linear1.bias, 0.0)
        # linear2 = nn.Linear(1024, 100)
        # nn.init.xavier_uniform_(linear2.weight.unsqueeze(0))
        # nn.init.xavier_uniform_(linear2.bias.unsqueeze(0))
        # self.param = nn.Sequential(
        #     linear1,
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     linear2
        # )
#         self.in_features = in_features
#         self.out_features = out_features
#         self.z_in_dim = z_in_dim
# #         self.bias = nn.Parameter(torch.zeros(100))
    
    def forward(self):
        param = self.hypernetwork[0](self.z[0])
        # param2 = self.hypernetwork[1](self.z[1])
        # param3 = self.hypernetwork[2](self.z[2])
        # param = torch.cat([param1, param2, param3], dim=0)
        # param = torch.transpose(param, 0, 1)
        # param = self.param(param)
        # param = torch.transpose(param, 0, 1)
        weight = param[:, :-1]
        bias = param[:, -1]
        return  Parameter(weight), Parameter(bias)
    
class HyperConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        z_in_dim: int,
        z_out_dim: int,
        bias: bool=True,
    ) -> None:
        super(HyperConv, self).__init__()
        self.hypernetwork = nn.Sequential(
            HyperConv_Module(in_features, out_features, 512, 2048),
            HyperConv_Module(in_features, out_features, 512, 512),
            HyperConv_Module(in_features, out_features, 2048, 512),
        )
#         print(in_features, out_features, z_in_dim,z_out_dim)
        self.z = [None, None, None]
        linear1 = nn.Linear(3072, 1024)
        nn.init.kaiming_normal_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        linear2 = nn.Linear(1024, 100)
        nn.init.kaiming_normal_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)
        self.param = nn.Sequential(
            linear1,
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            linear2
        )
#         self.in_features = in_features
#         self.out_features = out_features
#         self.z_in_dim = z_in_dim
# #         self.bias = nn.Parameter(torch.zeros(100))
    
    def forward(self):
        param1 = self.hypernetwork[0](self.z[0])
        param2 = self.hypernetwork[1](self.z[1])
        param3 = self.hypernetwork[2](self.z[2])
        param = torch.cat([param1, param2, param3], dim=0)
        param2 = torch.transpose(param, 0, 1)
        param2 = self.param(param2)
        param3 = torch.transpose(param2, 0, 1)
        weight = param3[:, :-1]
        bias = param3[:, -1]
        return  weight, bias