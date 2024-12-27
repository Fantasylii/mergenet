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
from dataset.lr_scheduler import WarmUpLR
import logging
import torch.nn as nn
import torch.nn.functional as F
from model.MobileNet_v2 import mobilenetv2
from model.ResNet import resnet50
from model.param_attention import ParamAttention
from torch.optim.lr_scheduler import MultiStepLR

EPOCH_NUM = 200
best_acc_mbv = 0.0  #############
best_acc_res = 0.0 

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

logging.basicConfig(filename='',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(res):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Runnint at ", device)
    print(torch.cuda.get_device_name())
    res.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_res = optim.SGD(res.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = MultiStepLR(optimizer_res, milestones=[60, 120, 160], gamma=0.2)
    iter_per_epoch = len(train_dataloader)
    # warmup_scheduler = WarmUpLR(optimizer_res, iter_per_epoch)

    for epoch in range(EPOCH_NUM):
        res.train()
     
        loss_total_mbv = 0.0
        loss_total = 0.0
        print('epoch: %d | lr: %f'% (epoch, optimizer_res.param_groups[0]["lr"]))
        logger.info('epoch: %d | lr: %f'% (epoch, optimizer_res.param_groups[0]["lr"]))
        progress_bar = tqdm(train_dataloader,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)
        for (img, label)  in progress_bar:
            img, label = img.to(device), label.to(device)
            optimizer_res.zero_grad()
            out = res(img)
            loss = criterion(out, label)
            loss.backward()
            loss_total += loss.item()
            optimizer_res.step()
            
            # if epoch < 1:
            #     warmup_scheduler.step()
            progress_bar.set_postfix({'Loss_mbv': '{:.6f}'.format(loss.item()), 'lr': '{:.6f}'.format(optimizer_res.param_groups[0]['lr'])})

        lr_scheduler.step()
        tqdm.write(f'Epoch {epoch}')
        logger.info(f'\nEpoch {epoch}')
        loss_train_ave = loss_total/len(train_dataloader)

        tqdm.write(f'Res Training Loss: {loss_train_ave}')
        logger.info(f'Res Training Loss: {loss_train_ave}')
        top1_acc, top5_acc = test(res, device)
        tqdm.write(f'Res Top1 Acc: {top1_acc}, Res Top5 Acc: {top5_acc}')
        logger.info(f'Res Top1 Acc: {top1_acc}, Res Top5 Acc: {top5_acc}')
      
    logger.info('%f'% (best_acc_res))
  
def test(model, device):
    global best_acc_res
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
        if top1_acc / total > best_acc_res:
            best_acc_res = top1_acc / total
            torch.save(model.state_dict(), '')
    return 100. * top1_acc / total, 100. * top5_acc / total
        
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


    logger.info(f'Linear -> Linear')
    
    start = time.time()
    logger.info('Start write!!!\n')

    res = mobilenetv2()
    print(f"Resnet 50: {sum(p.numel() for p in res.parameters())}")
    train(res)

if __name__ == '__main__':
    res = mobilenetv2()
    main()