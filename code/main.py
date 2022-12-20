
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
#from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SwinCIR
from test import test
from train import train
from data import TrainDataset, TestDataset

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

### define global variables
crop_out = 8
#epochs = 1158 ### around 500,000 iterations for batch=8
epochs = 579   ### around 500,000 iterations for batch=4

div2k_img_path = '/mnt/home/20160788/data/DIV2K_train_LR_bicubic/X4'
div2k_lbl_path = '/mnt/home/20160788/data/DIV2K_train_HR'

flickr2k_img_path = '/mnt/home/20160788/data/Flickr2K/Flickr2K_LR_bicubic/X4'
flickr2k_lbl_path = '/mnt/home/20160788/data/Flickr2K/Flickr2K_HR'

valid_img_path = '/mnt/home/20160788/data/DIV2K_valid_LR_bicubic/X4'
valid_lbl_path = '/mnt/home/20160788/data/DIV2K_valid_HR'

set5_img_path = '/mnt/home/20160788/data/Set5_LR'
set5_lbl_path = '/mnt/home/20160788/data/Set5_HR'
set14_img_path = '/mnt/home/20160788/data/Set14_LR'
set14_lbl_path = '/mnt/home/20160788/data/Set14_HR'
urban100_img_path = '/mnt/home/20160788/data/Urban100_LR'
urban100_lbl_path = '/mnt/home/20160788/data/Urban100_HR'

last_pnt_path = '../last.pt'
old_pnt_path = '../old.pt'
check_pnt_path = '../best.pt'
log_path = '../logdir'

### make log directory
if not os.path.exists(log_path):
    os.makedirs(log_path)

### analyze suffix
resume = (len(sys.argv) > 1 and (sys.argv[1] == '-r' or sys.argv[1] == '-R' or sys.argv[1] == '-resume'))
check_param = (len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '-P' or sys.argv[1] == '-param'))

### define model
net = SwinCIR()

### check parameter num
if check_param:
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    quit()

### define tensorboard writer
writer = SummaryWriter(log_path)

### define datasets
div2k_dataset = TrainDataset(div2k_img_path, div2k_lbl_path, crop_size=64, do_rand=True)
flickr2k_dataset = TrainDataset(flickr2k_img_path, flickr2k_lbl_path, crop_size=64, do_rand=True)
valid_dataset = TrainDataset(valid_img_path, valid_lbl_path, do_rand=False)
set5_dataset = TestDataset(set5_img_path, set5_lbl_path)
set14_dataset = TestDataset(set14_img_path, set14_lbl_path)
urban100_dataset = TestDataset(urban100_img_path, urban100_lbl_path)

### setup for train and do training
def setup_and_train(rank, ngpus_per_node, net):
    device = 2+rank
    torch.cuda.set_device(device)
    
    # initialize miulti-GPU process
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
    
    # send network to device
    net = DistributedDataParallel(net.to(device), device_ids=[device])

    # define train dataloader
    div2k_sampler = DistributedSampler(div2k_dataset)
    flickr2k_sampler = DistributedSampler(flickr2k_dataset)
    div2k_dataloader = DataLoader(div2k_dataset, batch_size=2, num_workers=2, sampler=div2k_sampler, pin_memory=True)
    flickr2k_dataloader = DataLoader(flickr2k_dataset, batch_size=2, num_workers=2, sampler=flickr2k_sampler, pin_memory=True)

    # define validation dataloader
    valid_sampler = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, sampler=valid_sampler, pin_memory=True)

    # define train variables
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99))
    #scheduler = MultiStepLR(optimizer, list(map(int, [0.5*epochs, 0.8*epochs, 0.9*epochs, 0.95*epochs])), 0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # make args
    args = {
        'net': net,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'device': device,
        'crop_out': crop_out,
        'epochs': epochs,
        'train_dataloaders': [div2k_dataloader, flickr2k_dataloader],
        'valid_dataloader': valid_dataloader,
        'check_pnt_path': check_pnt_path,
        'last_pnt_path': last_pnt_path,
        'old_pnt_path': old_pnt_path,
        'writer': writer
    }

    # do train
    train(args, resume)

def setup_and_test(rank, net):
    device = rank

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=rank)

    net = DistributedDataParallel(net.to(device), device_ids=[device])

    # define test dataloaders
    set5_sampler = DistributedSampler(set5_dataset)
    set14_sampler = DistributedSampler(set14_dataset)
    urban100_sampler = DistributedSampler(urban100_dataset)
    set5_dataloader = DataLoader(set5_dataset, batch_size=1, sampler=set5_sampler, pin_memory=True)
    set14_dataloader = DataLoader(set14_dataset, batch_size=1, sampler=set14_sampler, pin_memory=True)
    urban100_dataloader = DataLoader(urban100_dataset, batch_size=1, sampler=urban100_sampler, pin_memory=True)
    
    # make args
    args = {
        'net': net.to(device),
        'device': device,
        'crop_out': crop_out,
        'check_pnt_path': check_pnt_path,
        'writer': writer
    }

    # do test
    args['test_dataloader'] = set5_dataloader
    test(args, 'Set5')

    args['test_dataloader'] = set14_dataloader
    test(args, 'Set14')

    args['test_dataloader'] = urban100_dataloader
    test(args, 'Urban100')


if __name__ == '__main__':
    ### train
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8967'
    #ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node=2
    mp.spawn(setup_and_train, nprocs=ngpus_per_node, args=(ngpus_per_node, net))

    ### test
    mp.spawn(setup_and_test, nprocs=1, args=(net,))


