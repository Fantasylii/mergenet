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
from dataset.cls_dataloader import train_dataloader, test_dataloader
import logging
import torch.nn as nn
import torch.nn.functional as F
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

LR = 0.001
EPOCH_NUM = 200
best_acc_mbv = 0.0  #############
best_acc_res = 0.0 

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

logging.basicConfig(filename='/data/likunxi/attention/log_0213.txt',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def stats_params(model, weight_decay=5e-4):
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

def hypernetwork_update(model, param, final_param, optimizer, lr_scheduler, epoch):
    # print(param.size(), final_param.size())
    optimizer.zero_grad()
    delta_theta = param - final_param
    hn_grads = torch.autograd.grad(
        [param], model.parameters(), grad_outputs=delta_theta, allow_unused=True
    )

    # update hypernetwork weights
    for p, g in zip(model.parameters(), hn_grads):
        p.grad = g
        # if g is None:
        #     print(22222222222222, name)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
    optimizer.step()
    if epoch >= 4:
        lr_scheduler.step()

def train(mbv2, res, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Runnint at ", device)
    print(torch.cuda.get_device_name())
    param_attention_l = ParamAttention(config, mode='a')
    param_attention_l.to(device)
    # param_attention_r = ParamAttention(config, mode='b')
    # param_attention_r.to(device)
    mbv2 = mbv2.to(device)
    res = res.to(device)
    f = config['f']
    param_atten_l = stats_params(param_attention_l)
    # param_atten_r = stats_params(param_attention_r, weight_decay=1e-2)
    # kl_div = nn.KLDivLoss(reduction='batchmean')
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_mbv = optim.SGD(mbv2.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler_mbv = MultiStepLR(optimizer_mbv, milestones=[60, 120, 160], gamma=0.2)
    optimizer_res = optim.SGD(res.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler_res = MultiStepLR(optimizer_res, milestones=[60, 120, 160], gamma=0.2)

    optimizer_atten_l = optim.AdamW(param_atten_l, lr=config['lr'])
    lr_scheduler_atten_l = CosineAnnealingLR(optimizer_atten_l, T_max=(EPOCH_NUM - 4) * len(train_dataloader) // f)
    # optimizer_atten_r = optim.AdamW(param_atten_r, lr=0.001)
    # lr_scheduler_atten_r = CosineAnnealingLR(optimizer_atten_r, T_max=(EPOCH_NUM - 4) * len(train_dataloader) // 2)
 
    cnt = 0
    for epoch in range(EPOCH_NUM):
        param_attention_l.train()
        # param_attention_r.train()
        mbv2.train()
        res.train()
        loss_total_mbv = 0.0
        loss_total_res = 0.0
        print('epoch: %d | lr: %f'% (epoch, optimizer_mbv.param_groups[0]["lr"]))
        logger.info('epoch: %d | lr: %f'% (epoch, optimizer_mbv.param_groups[0]["lr"]))
        progress_bar = tqdm(train_dataloader,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        for data in progress_bar:
            param_a = {
                'conv': mbv2.stage6[2].residual[6].weight.data.clone().detach().requires_grad_(True).to(device),
                # 'conv_bias': mbv2.stage6[2].residual[6].bias.data.clone().detach().requires_grad_(True).to(device),
                # 'linear_weight': mbv2.model.classifier[1].weight.data.clone().detach().requires_grad_(True).to(device),
                # 'linear_bias': mbv2.model.classifier[1].bias.data.clone().detach().requires_grad_(True).to(device)
            }
            param_b = {
                # 'conv': res.model.layer4[0].conv1.A.data.clone().detach().requires_grad_(True).to(device),
                # 'conv2': res.model.layer4[2].conv2.weight.data.clone().detach().requires_grad_(True).to(device),
                # 'conv3': res.model.layer4[2].conv3.weight.data.clone().detach().requires_grad_(True).to(device),
                'linear_weight': res.fc.weight.data.clone().detach().requires_grad_(True).to(device),
            }
            if cnt % f == 0:
                    out_a = param_attention_l(param_a, param_b)
                    # make_dot(out_a, params={"param_a": param_a, "param_b": param_b})
                    new_param_mbv = {
                        'stage6.2.residual.6.weight': out_a
                        # 'stage6.2.residual.6.bias': out_a[:, -1].reshape(param_a['conv_bias'].size()),
                    }
                    mbv2.load_state_dict(new_param_mbv, strict=False)
                    # print(out_a[:, -1].reshape(param_a['conv_bias'].size())[:2],
                    #       mbv2.stage6[2].residual[6].bias[:2])
                    # out_b = param_attention_r(param_b, param_a)
                    # new_param_res = {'model.layer4.0.conv1.A': out_b}
                    # res.load_state_dict(new_param_res, strict=False)
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            optimizer_res.zero_grad()
            out = res(img)
            loss_res = criterion(out, label)
            loss_res.backward()
            loss_total_res += loss_res.item() 
            optimizer_res.step()
    
            img, label = data[0], data[1]
            img, label = img.to(device), label.to(device)
            # mbv2.model.classifier[1].hyper_module.z[0] = res.fc[0].weight.data.detach().to(device)
            optimizer_mbv.zero_grad()
            out = mbv2(img)
            loss_mbv = criterion2(out, label)
            loss_mbv.backward()
            
            # print(param_attention.layer[0].attention_x_x.fc_q.weight.grad)
            # print(mbv2.model.features[18][0].weight.grad)
            loss_total_mbv += loss_mbv.item()
            optimizer_mbv.step()
        
            #update hypernetwork patameters
            if cnt % f == 0:
                    final_state_mbv2 = mbv2.stage6[2].residual[6].weight
                    hypernetwork_update(param_attention_l, out_a, final_state_mbv2, optimizer_atten_l, lr_scheduler_atten_l, epoch)
                    # final_state_res = res.model.layer4[0].conv1.A
                    # hypernetwork_update(param_attention_r, out_b, final_state_res, optimizer_atten_r, lr_scheduler_atten_r, epoch)

            cnt += 1
            progress_bar.set_postfix({'Loss_mbv': '{:.6f}'.format(loss_mbv.item()), 'Loss_res': '{:.6f}'.format(loss_res.item()), 'lr': '{:.6f}'.format(optimizer_mbv.param_groups[0]['lr'])})

        lr_scheduler_mbv.step()
        lr_scheduler_res.step()
        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        loss_train_ave1 = loss_total_mbv/len(train_dataloader)
        loss_train_ave2 = loss_total_res/len(train_dataloader)
        tqdm.write(f'Mbv Training Loss: {loss_train_ave1}, Res Training Loss: {loss_train_ave2}')
        logger.info(f'Mbv Training Loss: {loss_train_ave1}, Res Training Loss: {loss_train_ave2}')
        # with torch.no_grad():
        #     for i in range(config['num_layers_a']):
        #         weights1 = F.softmax(param_attention_l.layer[i].weights, dim=-1)
        #         logger.info(weights1)
            # for i in range(config['num_layers_b']):
            #     weights1 = F.softmax(param_attention_r.layer[i].weights, dim=-1)
            #     logger.info(weights1)
        top1_acc, top5_acc, test_loss_mbv = test(mbv2, device, 'Mbv')
        tqdm.write(f'Mbv Top1 Acc: {top1_acc}, Mbv Top5 Acc: {top5_acc}')
        logger.info(f'Mbv Top1 Acc: {top1_acc}, Mbv Top5 Acc: {top5_acc}')
        top1_acc, top5_acc, test_loss_res = test(res, device, 'Res')
        tqdm.write(f'Res Top1 Acc: {top1_acc}, Res Top5 Acc: {top5_acc}')
        logger.info(f'Res Top1 Acc: {top1_acc}, Res Top5 Acc: {top5_acc}')
        tqdm.write(f'Mbv Test Loss: {test_loss_mbv}, Res Test Loss: {test_loss_res}')
        logger.info(f'Mbv Test Loss: {test_loss_mbv}, Res Test Loss: {test_loss_res}')
      
    logger.info('%f %f'% (best_acc_mbv, best_acc_res))
  
def test(model, device, model_name):
    global best_acc_mbv, best_acc_res
    model.eval()
    top1_acc = 0.
    top5_acc = 0.
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
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
    return 100. * top1_acc / total, 100. * top5_acc / total, running_loss / len(test_dataloader)
        
def main():
    warnings.filterwarnings("ignore")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
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

    config['a_size_conv'] = [160, 960]
    config['a_size_linear'] = [100, 1280]
    config['b_size_conv'] = [4, 1024]
    # config['b_size_conv2'] = [512, 512]
    # config['b_size_conv3'] = [2048, 512]
    config['b_size_linear'] = [100, 2048]
    layers = config['num_layers']
    
    config['mode'] = 5

    global best_acc_mbv, best_acc_res
    for i in [0.1, 0.01, 0.001, 0.0001]:
        for j in [1, 2, 3, 4, 5]:
            config['lr'] = i
            config['f'] = j
            best_acc_mbv, best_acc_res = 0., 0.
            mbv2 = mobilenetv2()
            res = resnet50()
            print(f"Mobilenet v2: {sum(p.numel() for p in mbv2.parameters())}")
            print(f"Resnet 50: {sum(p.numel() for p in res.parameters())}")
            logger.info(f'lr:{i}, f:{j}')
            train(mbv2, res, config)

if __name__ == '__main__':
    main()