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

logging.basicConfig(filename='/data/likunxi/attention/log_1125.txt',
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

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
    

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

def train(mbv2, res, config):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Runnint at ", device)
    print(torch.cuda.get_device_name())
    mbv2.to(device)
    res.to(device)
    top1_acc, top5_acc, test_loss_mbv = test(mbv2, device, 'Mbv')
    print(top1_acc, top5_acc)
    top1_acc, top5_acc, test_loss_mbv = test(res, device, 'Res')
    print(top1_acc, top5_acc)
    param_mbv = stats_params(mbv2)
    param_res = stats_params(res)
    # kl_div = nn.KLDivLoss(reduction='batchmean')
    dist_criterion = RkdDistance()
    angle_criterion = RKdAngle()
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_mbv = optim.AdamW(param_mbv, lr=0.01)
    lr_scheduler_mbv = CosineAnnealingLR(optimizer_mbv, T_max=(EPOCH_NUM - 4) * len(train_dataloader))
    optimizer_res = optim.AdamW(param_res, lr=0.01)
    lr_scheduler_res = CosineAnnealingLR(optimizer_res, T_max=(EPOCH_NUM - 4) * len(train_dataloader))

    for epoch in range(EPOCH_NUM):
        mbv2.train()
        res.eval()
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
                out_res = res(img)

            img, label = data1[0], data1[1]
            img, label = img.to(device), label.to(device)
            # mbv2.model.classifier[1].hyper_module.z[0] = res.fc[0].weight.data.detach().to(device)
            optimizer_mbv.zero_grad()
            out = mbv2(img)
            loss_mbv = criterion2(out, label)
            dist_loss = dist_criterion(out, out_res)
            angle_loss = angle_criterion(out, out_res)
            loss = loss_mbv + dist_loss + 2 * angle_loss
            loss.backward()
            # print(param_attention.layer[0].attention_x_x.fc_q.weight.grad)
            # print(mbv2.model.features[18][0].weight.grad)
            loss_total_mbv += loss.item()
            optimizer_mbv.step()
            if epoch >= 4:
                lr_scheduler_mbv.step()
    
            progress_bar.set_postfix({'Loss_mbv': '{:.6f}'.format(loss.item()), 'lr': '{:.6f}'.format(optimizer_mbv.param_groups[0]['lr'])})

        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        loss_train_ave1 = loss_total_mbv/len(train_dataloader)

        tqdm.write(f'Mbv Training Loss: {loss_train_ave1}')
        logger.info(f'Mbv Training Loss: {loss_train_ave1}')
        top1_acc, top5_acc, test_loss_mbv = test(mbv2, device, 'Mbv')
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
    mbv2.load_state_dict(torch.load('/data/likunxi/attention/checkpoint/mbv2_cifar100.pkl', map_location="cpu"))
    res = ResNet50(out_features=100, param_attention=True, pretrained=None)
    res.load_state_dict(torch.load('/data/likunxi/attention/checkpoint/res50_cifar100.pkl', map_location="cpu"))
    config['a_size_conv'] = [160, 960]
    config['a_size_linear'] = [100, 1280]
    config['b_size_conv'] = [512, 1024]
    # config['b_size_conv2'] = [512, 512]
    # config['b_size_conv3'] = [2048, 512]
    config['b_size_linear'] = [100, 2048]
    print(f"Mobilenet v2: {sum(p.numel() for p in mbv2.parameters())}")
    print(f"Resnet 50: {sum(p.numel() for p in res.parameters())}")
    train(mbv2, res, config)

if __name__ == '__main__':
    main()