import torch
import time
import yaml
from tqdm import tqdm
import random
import os
from torchviz import make_dot
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import math
from dataset.cls_dataloader import train_dataloader, test_loader_mbv, test_loader_res
import logging
import torch.nn as nn
import torch.nn.functional as F
from model.MobileNet_v2 import Moblienet_v2
from model.ResNet import ResNet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import CosineAnnealingLR

LR = 0.001
EPOCH_NUM = 100
best_acc_mbv = 0.0  #############
best_acc_res = 0.0 

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

logging.basicConfig(filename='/data/likunxi/attention/log_nkd.txt',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def stats_params(model, weight_decay=1.0e-5):
    params_without_wd = []
    params_with_wd = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
         
    param = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
    ]
    return param

class NKDLoss(nn.Module):

    """ PyTorch version of NKD """

    def __init__(self,
                 temp=1.0,
                 gamma=1.5,
                 ):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):
        
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)
        
        # N*class
        S_i = self.log_softmax(logit_s/self.temp)
        T_i = F.softmax(logit_t/self.temp, dim=1)     

        loss_non =  (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non 

def train(mbv2, res, config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Runnint at ", device)
    print(torch.cuda.get_device_name())
    mbv2.to(device)
    res.to(device)

    param_mbv = stats_params(mbv2)
    param_res = stats_params(res)
    # kl_div = nn.KLDivLoss(reduction='batchmean')
    nkd = NKDLoss()
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_mbv = optim.AdamW(param_mbv, lr=0.01)
    lr_scheduler_mbv = CosineAnnealingLR(optimizer_mbv, T_max=(EPOCH_NUM - 4) * len(train_dataloader))
    optimizer_res = optim.AdamW(param_res, lr=0.01)
    lr_scheduler_res = CosineAnnealingLR(optimizer_res, T_max=(EPOCH_NUM - 4) * len(train_dataloader))

    for epoch in range(EPOCH_NUM):
        mbv2.eval()
        res.train()
        loss_total_mbv = 0.0
        loss_total_res = 0.0
        print('epoch: %d | lr: %f'% (epoch, optimizer_mbv.param_groups[0]["lr"]))
        logger.info('epoch: %d | lr: %f'% (epoch, optimizer_mbv.param_groups[0]["lr"]))
        progress_bar = tqdm(train_dataloader,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        for (data1, data2) in progress_bar:
            img, label = data1[0], data1[1]
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                out_mbv = mbv2(img)

            img, label = data2[0], data2[1]
            img, label = img.to(device), label.to(device)
            # mbv2.model.classifier[1].hyper_module.z[0] = res.fc[0].weight.data.detach().to(device)
            optimizer_res.zero_grad()
            out = res(img)
          
            loss = nkd(out, out_mbv, label)
            loss.backward()
            # print(param_attention.layer[0].attention_x_x.fc_q.weight.grad)
            # print(mbv2.model.features[18][0].weight.grad)
            loss_total_res += loss.item()
            optimizer_res.step()
            if epoch >= 4:
                lr_scheduler_res.step()
    
            progress_bar.set_postfix({'Loss_mbv': '{:.6f}'.format(loss.item()), 'lr': '{:.6f}'.format(optimizer_res.param_groups[0]['lr'])})

        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        loss_train_ave1 = loss_total_mbv/len(train_dataloader)

        tqdm.write(f'Mbv Training Loss: {loss_train_ave1}')
        logger.info(f'Mbv Training Loss: {loss_train_ave1}')
        top1_acc, top5_acc, test_loss_mbv = test(res, device, 'Res')
        tqdm.write(f'Mbv Top1 Acc: {top1_acc}, Mbv Top5 Acc: {top5_acc}')
        logger.info(f'Mbv Top1 Acc: {top1_acc}, Mbv Top5 Acc: {top5_acc}')
      
    logger.info('%f %f'% (best_acc_mbv, best_acc_res))
  
def test(model, device, model_name):
    global best_acc_mbv, best_acc_res
    model.eval()
    top1_acc = 0.
    top5_acc = 0.
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    test_loader = test_loader_mbv if model_name == 'Mbv' else test_loader_res
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            out = model(img)
            
            loss = criterion(out, label)
            running_loss += loss.item()
            pred1 = out.argmax(dim=1)
            label_resize = label.view(-1,1)
            _, pred5 = out.topk(5, 1, True, True)
            total += label.size(0)
            top1_acc += pred1.eq(label).sum().item()
            top5_acc += torch.eq(pred5, label_resize).sum().float().item()
        if model_name == 'Mbv' and top1_acc / total > best_acc_mbv:
            best_acc_mbv = top1_acc / total
            torch.save(model.state_dict(), '/data/likunxi/attention/checkpoint/mbv2_cifar100.pkl')
        elif model_name == 'Res' and top1_acc / total > best_acc_res:
            best_acc_res = top1_acc / total
            torch.save(model.state_dict(), '/data/likunxi/attention/checkpoint/res50_cifar100.pkl')
    return 100. * top1_acc / total, 100. * top5_acc / total, running_loss / len(test_loader)
        
def main():
    warnings.filterwarnings("ignore")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    cpu_num = 4 
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    config = yaml.load(open('/data/likunxi/attention/config/param_attention_config.yaml', 'r'), Loader=yaml.Loader)

    logger.info(f'Linear -> Linear')
    
    start = time.time()
    logger.info('Start write!!!\n')

    mbv2 = Moblienet_v2(out_features=100, param_attention=True, pretrained=None)
    mbv2.load_state_dict(torch.load('/data/likunxi/attention/checkpoint/mbv2_base.pkl', map_location="cpu"))
    res = ResNet50(out_features=100, param_attention=True, pretrained=None)
    # res.load_state_dict(torch.load('/data/likunxi/attention/checkpoint/res50_base.pkl', map_location="cpu"))
    # config['a_size_conv'] = [160, 960]
    # config['a_size_linear'] = [100, 1280]
    # config['b_size_conv'] = [512, 1024]
    # # config['b_size_conv2'] = [512, 512]
    # # config['b_size_conv3'] = [2048, 512]
    # config['b_size_linear'] = [100, 2048]
    print(f"Mobilenet v2: {sum(p.numel() for p in mbv2.parameters())}")
    print(f"Resnet 50: {sum(p.numel() for p in res.parameters())}")
    train(mbv2, res, config)

if __name__ == '__main__':
    main()